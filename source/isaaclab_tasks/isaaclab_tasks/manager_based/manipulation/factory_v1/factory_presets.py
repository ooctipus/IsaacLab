# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PresetCfg definitions for Factory task-specific parameters.

Each preset maps task variant names to the corresponding keypoint-derived values.
The ``default`` field selects which variant is active when no CLI override is given.

Task categories (21 total):
  - ``nut_thread_m{4,8,12,16}`` — nut threading
  - ``gear_mesh_{small,medium,large}`` — gear meshing
  - ``rod_insert_{4,8,12,16}mm`` — round rod insertion
  - ``peg_insert_{4,8,12,16}mm`` — rectangular peg insertion
  - ``{usba,waterproof,bnc,dsub,rj45}`` — connector insertion

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
class GripperGraspOffsetCfg(PresetCfg):
    """Gripper grasp frame offset relative to the end-effector link per robot variant."""

    default: kpts.Offset = kpts.Offset()


@configclass
class JointEffortNamesCfg(PresetCfg):
    """Joint name regex for the effort penalty per robot variant."""

    default: str | None = None


# ---------------------------------------------------------------------------
# Task-specific presets
# ---------------------------------------------------------------------------


@configclass
class FixedAssetMapCfg(PresetCfg):
    """Mapping from scene entity key to :class:`NistBoardKeyPointsCfg` attribute name."""

    # Nut threading
    nut_thread_m4: dict = dict(fixed_asset="bolt_m4")
    nut_thread_m8: dict = dict(fixed_asset="bolt_m8")
    nut_thread_m12: dict = dict(fixed_asset="bolt_m12")
    nut_thread_m16: dict = dict(fixed_asset="bolt_m16")

    # Gear mesh — non-held gears are placed on the board as extra scene entities
    gear_mesh_small: dict = dict(fixed_asset="gear_base", medium_gear="medium_gear", large_gear="large_gear")
    gear_mesh_medium: dict = dict(fixed_asset="gear_base", small_gear="small_gear", large_gear="large_gear")
    gear_mesh_large: dict = dict(fixed_asset="gear_base", small_gear="small_gear", medium_gear="medium_gear")

    # Rod insert (round)
    rod_insert_4mm: dict = dict(fixed_asset="hole_4mm")
    rod_insert_8mm: dict = dict(fixed_asset="hole_8mm")
    rod_insert_12mm: dict = dict(fixed_asset="hole_12mm")
    rod_insert_16mm: dict = dict(fixed_asset="hole_16mm")

    # Peg insert (rectangular)
    peg_insert_4mm: dict = dict(fixed_asset="rectangular_hole_4mm")
    peg_insert_8mm: dict = dict(fixed_asset="rectangular_hole_8mm")
    peg_insert_12mm: dict = dict(fixed_asset="rectangular_hole_12mm")
    peg_insert_16mm: dict = dict(fixed_asset="rectangular_hole_16mm")

    # Connector insert
    usba: dict = dict(fixed_asset="usba_socket")
    waterproof: dict = dict(fixed_asset="waterproof_socket")
    bnc: dict = dict(fixed_asset="bnc_socket")
    dsub: dict = dict(fixed_asset="dsub_socket")
    rj45: dict = dict(fixed_asset="rj45_socket")

    default: dict = nut_thread_m16


@configclass
class HeldAssetTipCfg(PresetCfg):
    # Nut threading — tip of the bolt shaft where the nut enters
    nut_thread_m4: kpts.Offset = kpts.BOLT_M4_KEY_POINTS.bolt_tip_offset
    nut_thread_m8: kpts.Offset = kpts.BOLT_M8_KEY_POINTS.bolt_tip_offset
    nut_thread_m12: kpts.Offset = kpts.BOLT_M12_KEY_POINTS.bolt_tip_offset
    nut_thread_m16: kpts.Offset = kpts.BOLT_M16_KEY_POINTS.bolt_tip_offset

    # Gear mesh — tip of the gear shaft on the base
    gear_mesh_small: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.small_gear_tip_offset
    gear_mesh_medium: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.medium_gear_tip_offset
    gear_mesh_large: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.large_gear_tip_offset

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.KEY_POINTS_HOLE_4MM.hole_tip_offset
    rod_insert_8mm: kpts.Offset = kpts.KEY_POINTS_HOLE_8MM.hole_tip_offset
    rod_insert_12mm: kpts.Offset = kpts.KEY_POINTS_HOLE_12MM.hole_tip_offset
    rod_insert_16mm: kpts.Offset = kpts.KEY_POINTS_HOLE_16MM.hole_tip_offset

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_4MM.hole_tip_offset
    peg_insert_8mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_8MM.hole_tip_offset
    peg_insert_12mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_12MM.hole_tip_offset
    peg_insert_16mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_16MM.hole_tip_offset

    # Connector insert
    usba: kpts.Offset = kpts.KEY_POINTS_USB_A_SOCKET.entry
    waterproof: kpts.Offset = kpts.KEY_POINTS_WATERPROOF_SOCKET.entry
    bnc: kpts.Offset = kpts.KEY_POINTS_BNC_SOCKET.plug_assembled
    dsub: kpts.Offset = kpts.KEY_POINTS_D_SUB_SOCKET.entry
    rj45: kpts.Offset = kpts.KEY_POINTS_RJ45_SOCKET.entry

    default: kpts.Offset = nut_thread_m16


@configclass
class FixedAssetTipCfg(PresetCfg):
    # Nut threading
    nut_thread_m4: kpts.Offset = kpts.BOLT_M4_KEY_POINTS.bolt_tip_offset
    nut_thread_m8: kpts.Offset = kpts.BOLT_M8_KEY_POINTS.bolt_tip_offset
    nut_thread_m12: kpts.Offset = kpts.BOLT_M12_KEY_POINTS.bolt_tip_offset
    nut_thread_m16: kpts.Offset = kpts.BOLT_M16_KEY_POINTS.bolt_tip_offset

    # Gear mesh
    gear_mesh_small: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.small_gear_tip_offset
    gear_mesh_medium: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.medium_gear_tip_offset
    gear_mesh_large: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.large_gear_tip_offset

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.KEY_POINTS_HOLE_4MM.hole_tip_offset
    rod_insert_8mm: kpts.Offset = kpts.KEY_POINTS_HOLE_8MM.hole_tip_offset
    rod_insert_12mm: kpts.Offset = kpts.KEY_POINTS_HOLE_12MM.hole_tip_offset
    rod_insert_16mm: kpts.Offset = kpts.KEY_POINTS_HOLE_16MM.hole_tip_offset

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_4MM.hole_tip_offset
    peg_insert_8mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_8MM.hole_tip_offset
    peg_insert_12mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_12MM.hole_tip_offset
    peg_insert_16mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_16MM.hole_tip_offset

    # Connector insert
    usba: kpts.Offset = kpts.KEY_POINTS_USB_A_SOCKET.entry
    waterproof: kpts.Offset = kpts.KEY_POINTS_WATERPROOF_SOCKET.entry
    bnc: kpts.Offset = kpts.KEY_POINTS_BNC_SOCKET.plug_assembled
    dsub: kpts.Offset = kpts.KEY_POINTS_D_SUB_SOCKET.entry
    rj45: kpts.Offset = kpts.KEY_POINTS_RJ45_SOCKET.entry

    default: kpts.Offset = nut_thread_m16


@configclass
class AssembledOffsetCfg(PresetCfg):
    # Nut threading — where the nut rests when fully screwed
    nut_thread_m4: kpts.Offset = kpts.BOLT_M4_KEY_POINTS.fully_screwed_nut_offset
    nut_thread_m8: kpts.Offset = kpts.BOLT_M8_KEY_POINTS.fully_screwed_nut_offset
    nut_thread_m12: kpts.Offset = kpts.BOLT_M12_KEY_POINTS.fully_screwed_nut_offset
    nut_thread_m16: kpts.Offset = kpts.BOLT_M16_KEY_POINTS.fully_screwed_nut_offset

    # Gear mesh — bottom of the gear shaft where gear rests when assembled
    gear_mesh_small: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.small_gear_assembled_bottom_offset
    gear_mesh_medium: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.medium_gear_assembled_bottom_offset
    gear_mesh_large: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.large_gear_assembled_bottom_offset

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.KEY_POINTS_HOLE_4MM.inserted_peg_base_offset
    rod_insert_8mm: kpts.Offset = kpts.KEY_POINTS_HOLE_8MM.inserted_peg_base_offset
    rod_insert_12mm: kpts.Offset = kpts.KEY_POINTS_HOLE_12MM.inserted_peg_base_offset
    rod_insert_16mm: kpts.Offset = kpts.KEY_POINTS_HOLE_16MM.inserted_peg_base_offset

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_4MM.inserted_peg_base_offset
    peg_insert_8mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_8MM.inserted_peg_base_offset
    peg_insert_12mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_12MM.inserted_peg_base_offset
    peg_insert_16mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_16MM.inserted_peg_base_offset

    # Connector insert
    usba: kpts.Offset = kpts.KEY_POINTS_USB_A_SOCKET.plug_assembled
    waterproof: kpts.Offset = kpts.KEY_POINTS_WATERPROOF_SOCKET.plug_assembled
    bnc: kpts.Offset = kpts.KEY_POINTS_BNC_SOCKET.plug_assembled
    dsub: kpts.Offset = kpts.KEY_POINTS_D_SUB_SOCKET.plug_assembled
    rj45: kpts.Offset = kpts.KEY_POINTS_RJ45_SOCKET.plug_assembled

    default: kpts.Offset = nut_thread_m16


@configclass
class EntryOffsetCfg(PresetCfg):
    # Nut threading
    nut_thread_m4: kpts.Offset = kpts.BOLT_M4_KEY_POINTS.bolt_tip_offset
    nut_thread_m8: kpts.Offset = kpts.BOLT_M8_KEY_POINTS.bolt_tip_offset
    nut_thread_m12: kpts.Offset = kpts.BOLT_M12_KEY_POINTS.bolt_tip_offset
    nut_thread_m16: kpts.Offset = kpts.BOLT_M16_KEY_POINTS.bolt_tip_offset

    # Gear mesh
    gear_mesh_small: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.small_gear_tip_offset
    gear_mesh_medium: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.medium_gear_tip_offset
    gear_mesh_large: kpts.Offset = kpts.KEY_POINTS_GEAR_BASE.large_gear_tip_offset

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.KEY_POINTS_HOLE_4MM.hole_tip_offset
    rod_insert_8mm: kpts.Offset = kpts.KEY_POINTS_HOLE_8MM.hole_tip_offset
    rod_insert_12mm: kpts.Offset = kpts.KEY_POINTS_HOLE_12MM.hole_tip_offset
    rod_insert_16mm: kpts.Offset = kpts.KEY_POINTS_HOLE_16MM.hole_tip_offset

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_4MM.hole_tip_offset
    peg_insert_8mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_8MM.hole_tip_offset
    peg_insert_12mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_12MM.hole_tip_offset
    peg_insert_16mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_HOLE_16MM.hole_tip_offset

    # Connector insert
    usba: kpts.Offset = kpts.KEY_POINTS_USB_A_SOCKET.entry
    waterproof: kpts.Offset = kpts.KEY_POINTS_WATERPROOF_SOCKET.entry
    bnc: kpts.Offset = kpts.KEY_POINTS_BNC_SOCKET.plug_assembled
    dsub: kpts.Offset = kpts.KEY_POINTS_D_SUB_SOCKET.entry
    rj45: kpts.Offset = kpts.KEY_POINTS_RJ45_SOCKET.entry

    default: kpts.Offset = nut_thread_m16


@configclass
class HeldAssetAlignOffsetCfg(PresetCfg):
    # Nut threading — bottom of nut center axis for alignment
    nut_thread_m4: kpts.Offset = kpts.NUT_M4_KEY_POINTS.center_axis_bottom
    nut_thread_m8: kpts.Offset = kpts.NUT_M8_KEY_POINTS.center_axis_bottom
    nut_thread_m12: kpts.Offset = kpts.NUT_M12_KEY_POINTS.center_axis_bottom
    nut_thread_m16: kpts.Offset = kpts.NUT_M16_KEY_POINTS.center_axis_bottom

    # Gear mesh
    gear_mesh_small: kpts.Offset = kpts.KEY_POINTS_SMALL_GEAR.center_axis_bottom
    gear_mesh_medium: kpts.Offset = kpts.KEY_POINTS_MEDIUM_GEAR.center_axis_bottom
    gear_mesh_large: kpts.Offset = kpts.KEY_POINTS_LARGE_GEAR.center_axis_bottom

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.KEY_POINTS_ROD_4MM.center_axis_bottom
    rod_insert_8mm: kpts.Offset = kpts.KEY_POINTS_ROD_8MM.center_axis_bottom
    rod_insert_12mm: kpts.Offset = kpts.KEY_POINTS_ROD_12MM.center_axis_bottom
    rod_insert_16mm: kpts.Offset = kpts.KEY_POINTS_ROD_16MM.center_axis_bottom

    # Peg insert (rectangular) — peg tip is the alignment reference
    peg_insert_4mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_4MM.peg_tip
    peg_insert_8mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_8MM.peg_tip
    peg_insert_12mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_12MM.peg_tip
    peg_insert_16mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_16MM.peg_tip

    # Connector insert — insertion tip of the plug
    usba: kpts.Offset = kpts.KEY_POINTS_USB_A_PLUG.insertion_tip
    waterproof: kpts.Offset = kpts.KEY_POINTS_WATERPROOF_PLUG.insertion_tip
    bnc: kpts.Offset = kpts.KEY_POINTS_BNC_PLUG.insertion_tip
    dsub: kpts.Offset = kpts.KEY_POINTS_D_SUB_PLUG.insertion_tip
    rj45: kpts.Offset = kpts.KEY_POINTS_RJ45_PLUG.insertion_tip

    default: kpts.Offset = nut_thread_m16


@configclass
class HeldAssetGraspPointCfg(PresetCfg):
    # Nut threading
    nut_thread_m4: kpts.Offset = kpts.NUT_M4_KEY_POINTS.grasp_point
    nut_thread_m8: kpts.Offset = kpts.NUT_M8_KEY_POINTS.grasp_point
    nut_thread_m12: kpts.Offset = kpts.NUT_M12_KEY_POINTS.grasp_point
    nut_thread_m16: kpts.Offset = kpts.NUT_M16_KEY_POINTS.grasp_point

    # Gear mesh
    gear_mesh_small: kpts.Offset = kpts.KEY_POINTS_SMALL_GEAR.grasp_point
    gear_mesh_medium: kpts.Offset = kpts.KEY_POINTS_MEDIUM_GEAR.grasp_point
    gear_mesh_large: kpts.Offset = kpts.KEY_POINTS_LARGE_GEAR.grasp_point

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.KEY_POINTS_ROD_4MM.grasp_point
    rod_insert_8mm: kpts.Offset = kpts.KEY_POINTS_ROD_8MM.grasp_point
    rod_insert_12mm: kpts.Offset = kpts.KEY_POINTS_ROD_12MM.grasp_point
    rod_insert_16mm: kpts.Offset = kpts.KEY_POINTS_ROD_16MM.grasp_point

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_4MM.grasp_point
    peg_insert_8mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_8MM.grasp_point
    peg_insert_12mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_12MM.grasp_point
    peg_insert_16mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_16MM.grasp_point

    # Connector insert
    usba: kpts.Offset = kpts.KEY_POINTS_USB_A_PLUG.grasp_point
    waterproof: kpts.Offset = kpts.KEY_POINTS_WATERPROOF_PLUG.grasp_point
    bnc: kpts.Offset = kpts.KEY_POINTS_BNC_PLUG.grasp_point
    dsub: kpts.Offset = kpts.KEY_POINTS_D_SUB_PLUG.grasp_point
    rj45: kpts.Offset = kpts.KEY_POINTS_RJ45_PLUG.grasp_point

    default: kpts.Offset = nut_thread_m16


@configclass
class HeldAssetGraspDiameterCfg(PresetCfg):
    # Nut threading
    nut_thread_m4: float = kpts.NUT_M4_KEY_POINTS.grasp_diameter
    nut_thread_m8: float = kpts.NUT_M8_KEY_POINTS.grasp_diameter
    nut_thread_m12: float = kpts.NUT_M12_KEY_POINTS.grasp_diameter
    nut_thread_m16: float = kpts.NUT_M16_KEY_POINTS.grasp_diameter

    # Gear mesh
    gear_mesh_small: float = kpts.KEY_POINTS_SMALL_GEAR.grasp_diameter
    gear_mesh_medium: float = kpts.KEY_POINTS_MEDIUM_GEAR.grasp_diameter
    gear_mesh_large: float = kpts.KEY_POINTS_LARGE_GEAR.grasp_diameter

    # Rod insert (round)
    rod_insert_4mm: float = kpts.KEY_POINTS_ROD_4MM.grasp_diameter
    rod_insert_8mm: float = kpts.KEY_POINTS_ROD_8MM.grasp_diameter
    rod_insert_12mm: float = kpts.KEY_POINTS_ROD_12MM.grasp_diameter
    rod_insert_16mm: float = kpts.KEY_POINTS_ROD_16MM.grasp_diameter

    # Peg insert (rectangular)
    peg_insert_4mm: float = kpts.KEY_POINTS_RECTANGULAR_PEG_4MM.grasp_diameter
    peg_insert_8mm: float = kpts.KEY_POINTS_RECTANGULAR_PEG_8MM.grasp_diameter
    peg_insert_12mm: float = kpts.KEY_POINTS_RECTANGULAR_PEG_12MM.grasp_diameter
    peg_insert_16mm: float = kpts.KEY_POINTS_RECTANGULAR_PEG_16MM.grasp_diameter

    # Connector insert
    usba: float = kpts.KEY_POINTS_USB_A_PLUG.grasp_diameter
    waterproof: float = kpts.KEY_POINTS_WATERPROOF_PLUG.grasp_diameter
    bnc: float = kpts.KEY_POINTS_BNC_PLUG.grasp_diameter
    dsub: float = kpts.KEY_POINTS_D_SUB_PLUG.grasp_diameter
    rj45: float = kpts.KEY_POINTS_RJ45_PLUG.grasp_diameter

    default: float = nut_thread_m16


@configclass
class HeldAssetGraspMiddleCfg(PresetCfg):
    """Offset used for positioning the EE around the held asset.

    For nut variants this is the center_axis_middle (grasp from above the threading axis),
    while for all other variants it is the grasp_point.
    """

    # Nut threading — center axis middle for threading approach
    nut_thread_m4: kpts.Offset = kpts.NUT_M4_KEY_POINTS.center_axis_middle
    nut_thread_m8: kpts.Offset = kpts.NUT_M8_KEY_POINTS.center_axis_middle
    nut_thread_m12: kpts.Offset = kpts.NUT_M12_KEY_POINTS.center_axis_middle
    nut_thread_m16: kpts.Offset = kpts.NUT_M16_KEY_POINTS.center_axis_middle

    # Gear mesh
    gear_mesh_small: kpts.Offset = kpts.KEY_POINTS_SMALL_GEAR.grasp_point
    gear_mesh_medium: kpts.Offset = kpts.KEY_POINTS_MEDIUM_GEAR.grasp_point
    gear_mesh_large: kpts.Offset = kpts.KEY_POINTS_LARGE_GEAR.grasp_point

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.KEY_POINTS_ROD_4MM.grasp_point
    rod_insert_8mm: kpts.Offset = kpts.KEY_POINTS_ROD_8MM.grasp_point
    rod_insert_12mm: kpts.Offset = kpts.KEY_POINTS_ROD_12MM.grasp_point
    rod_insert_16mm: kpts.Offset = kpts.KEY_POINTS_ROD_16MM.grasp_point

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_4MM.grasp_point
    peg_insert_8mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_8MM.grasp_point
    peg_insert_12mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_12MM.grasp_point
    peg_insert_16mm: kpts.Offset = kpts.KEY_POINTS_RECTANGULAR_PEG_16MM.grasp_point

    # Connector insert
    usba: kpts.Offset = kpts.KEY_POINTS_USB_A_PLUG.grasp_point
    waterproof: kpts.Offset = kpts.KEY_POINTS_WATERPROOF_PLUG.grasp_point
    bnc: kpts.Offset = kpts.KEY_POINTS_BNC_PLUG.grasp_point
    dsub: kpts.Offset = kpts.KEY_POINTS_D_SUB_PLUG.grasp_point
    rj45: kpts.Offset = kpts.KEY_POINTS_RJ45_PLUG.grasp_point

    default: kpts.Offset = nut_thread_m16


# Pose ranges reused across size variants within each category
_NUT_GRASPED_RANGE = dict(
    x=(-0.005, 0.005), y=(-0.005, 0.005), z=(0.00, 0.035),
    roll=(3.141, 3.141), pitch=(-0.5, 0.5), yaw=(-2.09, 2.09),
)
_GEAR_GRASPED_RANGE = dict(
    x=(-0.02, 0.02), y=(-0.02, 0.02), z=(0.035, 0.045),
    roll=(3.141, 3.141), pitch=(-0.5, 0.5), yaw=(-2.09, 2.09),
)
_INSERT_GRASPED_RANGE = dict(
    x=(-0.005, 0.005), y=(-0.005, 0.005), z=(0.047, 0.057),
    roll=(3.141, 3.141), pitch=(-0.5, 0.5), yaw=(-2.09, 2.09),
)


@configclass
class GraspedPoseRangeCfg(PresetCfg):
    """Pose range for the ``start_grasped_then_assembled`` reset strategy."""

    # Nut threading
    nut_thread_m4: dict = _NUT_GRASPED_RANGE
    nut_thread_m8: dict = _NUT_GRASPED_RANGE
    nut_thread_m12: dict = _NUT_GRASPED_RANGE
    nut_thread_m16: dict = _NUT_GRASPED_RANGE

    # Gear mesh
    gear_mesh_small: dict = _GEAR_GRASPED_RANGE
    gear_mesh_medium: dict = _GEAR_GRASPED_RANGE
    gear_mesh_large: dict = _GEAR_GRASPED_RANGE

    # Rod insert (round)
    rod_insert_4mm: dict = _INSERT_GRASPED_RANGE
    rod_insert_8mm: dict = _INSERT_GRASPED_RANGE
    rod_insert_12mm: dict = _INSERT_GRASPED_RANGE
    rod_insert_16mm: dict = _INSERT_GRASPED_RANGE

    # Peg insert (rectangular)
    peg_insert_4mm: dict = _INSERT_GRASPED_RANGE
    peg_insert_8mm: dict = _INSERT_GRASPED_RANGE
    peg_insert_12mm: dict = _INSERT_GRASPED_RANGE
    peg_insert_16mm: dict = _INSERT_GRASPED_RANGE

    # Connector insert
    usba: dict = _INSERT_GRASPED_RANGE
    waterproof: dict = _INSERT_GRASPED_RANGE
    bnc: dict = _INSERT_GRASPED_RANGE
    dsub: dict = _INSERT_GRASPED_RANGE
    rj45: dict = _INSERT_GRASPED_RANGE

    default: dict = nut_thread_m16


_NUT_PARTIAL = (0.4, 1.1)
_GEAR_PARTIAL = (0.3, 1.0)
_INSERT_PARTIAL = (0.0, 1.0)


@configclass
class AssemblyFractionPartialCfg(PresetCfg):
    """Assembly fraction range for the ``start_assembled`` strategy."""

    nut_thread_m4: tuple = _NUT_PARTIAL
    nut_thread_m8: tuple = _NUT_PARTIAL
    nut_thread_m12: tuple = _NUT_PARTIAL
    nut_thread_m16: tuple = _NUT_PARTIAL

    gear_mesh_small: tuple = _GEAR_PARTIAL
    gear_mesh_medium: tuple = _GEAR_PARTIAL
    gear_mesh_large: tuple = _GEAR_PARTIAL

    rod_insert_4mm: tuple = _INSERT_PARTIAL
    rod_insert_8mm: tuple = _INSERT_PARTIAL
    rod_insert_12mm: tuple = _INSERT_PARTIAL
    rod_insert_16mm: tuple = _INSERT_PARTIAL

    peg_insert_4mm: tuple = _INSERT_PARTIAL
    peg_insert_8mm: tuple = _INSERT_PARTIAL
    peg_insert_12mm: tuple = _INSERT_PARTIAL
    peg_insert_16mm: tuple = _INSERT_PARTIAL

    usba: tuple = _INSERT_PARTIAL
    waterproof: tuple = _INSERT_PARTIAL
    bnc: tuple = _INSERT_PARTIAL
    dsub: tuple = _INSERT_PARTIAL
    rj45: tuple = _INSERT_PARTIAL

    default: tuple = nut_thread_m16


_NUT_FULL = (0.05, 0.5)
_GEAR_FULL = (0.1, 0.5)
_INSERT_FULL = (0.0, 0.5)


@configclass
class AssemblyFractionFullCfg(PresetCfg):
    """Assembly fraction range for the ``start_fully_assembled`` strategy."""

    nut_thread_m4: tuple = _NUT_FULL
    nut_thread_m8: tuple = _NUT_FULL
    nut_thread_m12: tuple = _NUT_FULL
    nut_thread_m16: tuple = _NUT_FULL

    gear_mesh_small: tuple = _GEAR_FULL
    gear_mesh_medium: tuple = _GEAR_FULL
    gear_mesh_large: tuple = _GEAR_FULL

    rod_insert_4mm: tuple = _INSERT_FULL
    rod_insert_8mm: tuple = _INSERT_FULL
    rod_insert_12mm: tuple = _INSERT_FULL
    rod_insert_16mm: tuple = _INSERT_FULL

    peg_insert_4mm: tuple = _INSERT_FULL
    peg_insert_8mm: tuple = _INSERT_FULL
    peg_insert_12mm: tuple = _INSERT_FULL
    peg_insert_16mm: tuple = _INSERT_FULL

    usba: tuple = _INSERT_FULL
    waterproof: tuple = _INSERT_FULL
    bnc: tuple = _INSERT_FULL
    dsub: tuple = _INSERT_FULL
    rj45: tuple = _INSERT_FULL

    default: tuple = nut_thread_m16


_ZERO_RATIO = (0.0, 0.0, 0.0)


@configclass
class AssemblyRatioCfg(PresetCfg):
    """Assembly ratio (linear displacement per radian of rotation) [m/rad]."""

    nut_thread_m4: tuple = (0.0, 0.0, kpts.NUT_M4_KEY_POINTS.screw_ratio / 6.2832)
    nut_thread_m8: tuple = (0.0, 0.0, kpts.NUT_M8_KEY_POINTS.screw_ratio / 6.2832)
    nut_thread_m12: tuple = (0.0, 0.0, kpts.NUT_M12_KEY_POINTS.screw_ratio / 6.2832)
    nut_thread_m16: tuple = (0.0, 0.0, kpts.NUT_M16_KEY_POINTS.screw_ratio / 6.2832)

    gear_mesh_small: tuple = _ZERO_RATIO
    gear_mesh_medium: tuple = _ZERO_RATIO
    gear_mesh_large: tuple = _ZERO_RATIO

    rod_insert_4mm: tuple = _ZERO_RATIO
    rod_insert_8mm: tuple = _ZERO_RATIO
    rod_insert_12mm: tuple = _ZERO_RATIO
    rod_insert_16mm: tuple = _ZERO_RATIO

    peg_insert_4mm: tuple = _ZERO_RATIO
    peg_insert_8mm: tuple = _ZERO_RATIO
    peg_insert_12mm: tuple = _ZERO_RATIO
    peg_insert_16mm: tuple = _ZERO_RATIO

    usba: tuple = _ZERO_RATIO
    waterproof: tuple = _ZERO_RATIO
    bnc: tuple = _ZERO_RATIO
    dsub: tuple = _ZERO_RATIO
    rj45: tuple = _ZERO_RATIO

    default: tuple = nut_thread_m16
