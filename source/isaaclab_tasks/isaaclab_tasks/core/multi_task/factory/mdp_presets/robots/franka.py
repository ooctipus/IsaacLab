# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda robot preset. Activate with ``presets=franka``."""

from __future__ import annotations

__all__: list[str] = []

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import preset

from ... import mdp
from ...assembly_keypoints import PANDA_HAND
from ...factory_assets_cfg import FRANKA_PANDA_NEWTON_CFG, FRANKA_PANDA_PHYSX_CFG
from ...factory_presets import (
    EndEffectorBodyCfg,
    FingerBodyNamesCfg,
    GripperBodyNamesCfg,
    GripperGraspOffsetCfg,
    GripperJointNamesCfg,
    IKJointNamesCfg,
    JointEffortNamesCfg,
)
from .robot_presets import (
    RobotActionsCfg,
    RobotArticulationCfg,
    RobotContactSensorsCfg,
)

# ---------------------------------------------------------------------------
# Robot identity / DOF naming
# ---------------------------------------------------------------------------

EndEffectorBodyCfg.franka = "panda_fingertip_centered"
GripperJointNamesCfg.franka = ["panda_finger.*"]
FingerBodyNamesCfg.franka = ["panda_leftfinger", "panda_rightfinger"]
GripperBodyNamesCfg.franka = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
IKJointNamesCfg.franka = ["panda_joint.*"]
GripperGraspOffsetCfg.franka = PANDA_HAND.gripper_center_grasp_point
JointEffortNamesCfg.franka = "(?!panda_joint7$|panda_finger_.*$).*"

# ---------------------------------------------------------------------------
# Articulation
# ---------------------------------------------------------------------------

# Backend-unification split: pick the PhysX or Newton/MuJoCo articulation cfg from the active
# backend preset. ``default`` and the compatibility ``physx`` selector keep PhysX behavior.
_FRANKA_PHYSX_CFG = FRANKA_PANDA_PHYSX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
RobotArticulationCfg.franka = preset(
    default=_FRANKA_PHYSX_CFG,
    isaacsim_physx=_FRANKA_PHYSX_CFG,
    physx=_FRANKA_PHYSX_CFG,
    newton_mjwarp=FRANKA_PANDA_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class FrankaActionsCfg:
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.02,
        use_zero_offset=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


RobotActionsCfg.franka = FrankaActionsCfg()


# ---------------------------------------------------------------------------
# Robot-specific scene sensors and rewards
# ---------------------------------------------------------------------------

RobotContactSensorsCfg.franka = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3)
