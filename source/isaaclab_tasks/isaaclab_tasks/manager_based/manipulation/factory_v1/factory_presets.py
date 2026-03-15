# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PresetCfg definitions for Factory task-specific parameters.

Each preset maps task variant names (``nut_thread``, ``gear_mesh``, ``peg_insert``)
to the corresponding keypoint-derived values.  The ``default`` field selects which
variant is active when no CLI override is given.

Robot-specific presets (``EndEffectorBodyCfg``, ``JointEffortNamesCfg``) map robot
variant names (``franka``) to body/joint name strings.
"""

from isaaclab.utils import configclass
from isaaclab_tasks.utils import PresetCfg

from . import assembly_keypoints as kpts


# ---------------------------------------------------------------------------
# Robot-specific presets
# ---------------------------------------------------------------------------


@configclass
class EndEffectorBodyCfg(PresetCfg):
    """End-effector body name per robot variant."""

    default: str = "end_effector"


@configclass
class GripperJointNamesCfg(PresetCfg):
    """Joint name regex for gripper/finger joints per robot variant."""

    default: list[str] | None = None


@configclass
class IKJointNamesCfg(PresetCfg):
    """Joint name regex used by the IK solver in reset strategies."""

    default: list[str] | None = None


@configclass
class JointEffortNamesCfg(PresetCfg):
    """Joint name regex for the effort penalty per robot variant."""

    default: str | None = None


# ---------------------------------------------------------------------------
# Task-specific presets
# ---------------------------------------------------------------------------


@configclass
class FixedAssetMapCfg(PresetCfg):
    """Mapping from scene entity key to :class:`KeyPointsNistBoard` attribute name."""

    nut_thread: dict = dict(fixed_asset="bolt_m16")
    gear_mesh: dict = dict(gear_base="gear_base", large_gear="large_gear", small_gear="small_gear")
    peg_insert: dict = dict(fixed_asset="hole_8mm")
    default: dict = nut_thread


@configclass
class HeldAssetTipCfg(PresetCfg):
    nut_thread: kpts.Offset = kpts.KEYPOINTS_BOLTHALFM16.bolt_tip_offset
    gear_mesh: kpts.Offset = kpts.KEYPOINTS_GEARBASE.medium_gear_tip_offset
    peg_insert: kpts.Offset = kpts.KEYPOINTS_HOLE8MM.hole_tip_offset
    default: kpts.Offset = nut_thread


@configclass
class FixedAssetTipCfg(PresetCfg):
    nut_thread: kpts.Offset = kpts.KEYPOINTS_BOLTHALFM16.bolt_tip_offset
    gear_mesh: kpts.Offset = kpts.KEYPOINTS_GEARBASE.medium_gear_tip_offset
    peg_insert: kpts.Offset = kpts.KEYPOINTS_HOLE8MM.hole_tip_offset
    default: kpts.Offset = nut_thread


@configclass
class AssembledOffsetCfg(PresetCfg):
    nut_thread: kpts.Offset = kpts.KEYPOINTS_BOLTHALFM16.fully_screwed_nut_offset
    gear_mesh: kpts.Offset = kpts.KEYPOINTS_GEARBASE.medium_gear_assembled_bottom_offset
    peg_insert: kpts.Offset = kpts.KEYPOINTS_HOLE8MM.inserted_peg_base_offset
    default: kpts.Offset = nut_thread


@configclass
class EntryOffsetCfg(PresetCfg):
    nut_thread: kpts.Offset = kpts.KEYPOINTS_BOLTHALFM16.bolt_tip_offset
    gear_mesh: kpts.Offset = kpts.KEYPOINTS_GEARBASE.medium_gear_tip_offset
    peg_insert: kpts.Offset = kpts.KEYPOINTS_HOLE8MM.hole_tip_offset
    default: kpts.Offset = nut_thread


@configclass
class HeldAssetAlignOffsetCfg(PresetCfg):
    nut_thread: kpts.Offset = kpts.KEYPOINTS_NUTM16.center_axis_bottom
    gear_mesh: kpts.Offset = kpts.KEYPOINTS_MEDIUMGEAR.center_axis_bottom
    peg_insert: kpts.Offset = kpts.KEYPOINTS_PEG8MM.center_axis_bottom
    default: kpts.Offset = nut_thread


@configclass
class HeldAssetGraspPointCfg(PresetCfg):
    nut_thread: kpts.Offset = kpts.KEYPOINTS_NUTM16.grasp_point
    gear_mesh: kpts.Offset = kpts.KEYPOINTS_MEDIUMGEAR.grasp_point
    peg_insert: kpts.Offset = kpts.KEYPOINTS_PEG8MM.grasp_point
    default: kpts.Offset = nut_thread


@configclass
class HeldAssetGraspDiameterCfg(PresetCfg):
    nut_thread: float = kpts.KEYPOINTS_NUTM16.grasp_diameter
    gear_mesh: float = kpts.KEYPOINTS_MEDIUMGEAR.grasp_diameter
    peg_insert: float = kpts.KEYPOINTS_PEG8MM.grasp_diameter
    default: float = nut_thread


@configclass
class HeldAssetGraspMiddleCfg(PresetCfg):
    """Offset used for positioning the EE around the held asset.

    For nut_thread this is the center_axis_middle (grasp from above the threading axis),
    while for gear_mesh and peg_insert it is the grasp_point.
    """

    nut_thread: kpts.Offset = kpts.KEYPOINTS_NUTM16.center_axis_middle
    gear_mesh: kpts.Offset = kpts.KEYPOINTS_MEDIUMGEAR.grasp_point
    peg_insert: kpts.Offset = kpts.KEYPOINTS_PEG8MM.grasp_point
    default: kpts.Offset = nut_thread


@configclass
class GraspedPoseRangeCfg(PresetCfg):
    """Pose range for the ``start_grasped_then_assembled`` reset strategy."""

    nut_thread: dict = dict(
        x=(-0.005, 0.005), y=(-0.005, 0.005), z=(0.00, 0.035),
        roll=(3.141, 3.141), pitch=(-0.5, 0.5), yaw=(-2.09, 2.09),
    )
    gear_mesh: dict = dict(
        x=(-0.02, 0.02), y=(-0.02, 0.02), z=(0.035, 0.045),
        roll=(3.141, 3.141), pitch=(-0.5, 0.5), yaw=(-2.09, 2.09),
    )
    peg_insert: dict = dict(
        x=(-0.005, 0.005), y=(-0.005, 0.005), z=(0.047, 0.057),
        roll=(3.141, 3.141), pitch=(-0.5, 0.5), yaw=(-2.09, 2.09),
    )
    default: dict = nut_thread


@configclass
class AssemblyFractionPartialCfg(PresetCfg):
    """Assembly fraction range for the ``start_assembled`` strategy."""

    nut_thread: tuple = (0.4, 1.1)
    gear_mesh: tuple = (0.3, 1.)
    peg_insert: tuple = (0.0, 1.0)
    default: tuple = nut_thread


@configclass
class AssemblyFractionFullCfg(PresetCfg):
    """Assembly fraction range for the ``start_fully_assembled`` strategy."""

    nut_thread: tuple = (0.05, 0.5)
    gear_mesh: tuple = (0.1, 0.5)
    peg_insert: tuple = (0.0, 0.5)
    default: tuple = nut_thread


@configclass
class AssemblyRatioCfg(PresetCfg):
    """Assembly ratio (linear displacement per radian of rotation)."""

    nut_thread: tuple = (0., 0., kpts.KEYPOINTS_NUTM16.screw_ratio / 6.2832)
    gear_mesh: tuple = (0., 0., 0.)
    peg_insert: tuple = (0., 0., 0.)
    default: tuple = nut_thread
