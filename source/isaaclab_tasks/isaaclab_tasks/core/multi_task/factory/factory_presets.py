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

import math

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from ..utils.symmetry import AssetSymmetryCfg, AxisSymmetryCfg
from . import assembly_keypoints as kpts
from .assembly_profile_cfg import (
    AssemblyProfileCfg,
    EndPointsSegmentCfg,
    IncrementalSegmentCfg,
    SymmetryOrbitCfg,
    UniformPoseNoiseCfg,
)

# Per-held-asset rotational symmetry -- the SINGLE source of truth shared by the
# assembly sampler (SymmetryOrbitCfg spawns an equivalent target) and the success
# criterion (HeldAssetSymmetryCfg accepts any equivalent). Continuous yaw for the
# round threaded/insert/gear parts; N-fold for the rectangular pegs; none for the
# keyed connectors.
_SYM_CONTINUOUS: AssetSymmetryCfg = AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=0)])
_SYM_4FOLD: AssetSymmetryCfg = AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=4)])
_SYM_2FOLD: AssetSymmetryCfg = AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=2)])
_SYM_NONE: AssetSymmetryCfg = AssetSymmetryCfg(elements=[AxisSymmetryCfg(order=1)])


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
class FingerBodyNamesCfg(PresetCfg):
    """Finger body names carrying the pad contact points, per robot variant."""

    default: list[str] | None = None


@configclass
class GripperBodyNamesCfg(PresetCfg):
    """Gripper body names (hand + fingers) probed for collision, per robot variant."""

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
    nut_thread_m4: kpts.Offset = kpts.BOLT_M4.bolt_tip_offset
    nut_thread_m8: kpts.Offset = kpts.BOLT_M8.bolt_tip_offset
    nut_thread_m12: kpts.Offset = kpts.BOLT_M12.bolt_tip_offset
    nut_thread_m16: kpts.Offset = kpts.BOLT_M16.bolt_tip_offset

    # Gear mesh — tip of the gear shaft on the base
    gear_mesh_small: kpts.Offset = kpts.GEAR_BASE.small_gear_tip_offset
    gear_mesh_medium: kpts.Offset = kpts.GEAR_BASE.medium_gear_tip_offset
    gear_mesh_large: kpts.Offset = kpts.GEAR_BASE.large_gear_tip_offset

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.HOLE_4MM.hole_tip_offset
    rod_insert_8mm: kpts.Offset = kpts.HOLE_8MM.hole_tip_offset
    rod_insert_12mm: kpts.Offset = kpts.HOLE_12MM.hole_tip_offset
    rod_insert_16mm: kpts.Offset = kpts.HOLE_16MM.hole_tip_offset

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.RECTANGULAR_HOLE_4MM.hole_tip_offset
    peg_insert_8mm: kpts.Offset = kpts.RECTANGULAR_HOLE_8MM.hole_tip_offset
    peg_insert_12mm: kpts.Offset = kpts.RECTANGULAR_HOLE_12MM.hole_tip_offset
    peg_insert_16mm: kpts.Offset = kpts.RECTANGULAR_HOLE_16MM.hole_tip_offset

    # Connector insert
    usba: kpts.Offset = kpts.USB_A_SOCKET.entry
    waterproof: kpts.Offset = kpts.WATERPROOF_SOCKET.entry
    bnc: kpts.Offset = kpts.BNC_SOCKET.entry
    dsub: kpts.Offset = kpts.D_SUB_SOCKET.entry
    rj45: kpts.Offset = kpts.RJ45_SOCKET.entry

    default: kpts.Offset = nut_thread_m16


@configclass
class FixedAssetTipCfg(PresetCfg):
    # Nut threading
    nut_thread_m4: kpts.Offset = kpts.BOLT_M4.bolt_tip_offset
    nut_thread_m8: kpts.Offset = kpts.BOLT_M8.bolt_tip_offset
    nut_thread_m12: kpts.Offset = kpts.BOLT_M12.bolt_tip_offset
    nut_thread_m16: kpts.Offset = kpts.BOLT_M16.bolt_tip_offset

    # Gear mesh
    gear_mesh_small: kpts.Offset = kpts.GEAR_BASE.small_gear_tip_offset
    gear_mesh_medium: kpts.Offset = kpts.GEAR_BASE.medium_gear_tip_offset
    gear_mesh_large: kpts.Offset = kpts.GEAR_BASE.large_gear_tip_offset

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.HOLE_4MM.hole_tip_offset
    rod_insert_8mm: kpts.Offset = kpts.HOLE_8MM.hole_tip_offset
    rod_insert_12mm: kpts.Offset = kpts.HOLE_12MM.hole_tip_offset
    rod_insert_16mm: kpts.Offset = kpts.HOLE_16MM.hole_tip_offset

    # Peg insert (rectangular)
    peg_insert_4mm: kpts.Offset = kpts.RECTANGULAR_HOLE_4MM.hole_tip_offset
    peg_insert_8mm: kpts.Offset = kpts.RECTANGULAR_HOLE_8MM.hole_tip_offset
    peg_insert_12mm: kpts.Offset = kpts.RECTANGULAR_HOLE_12MM.hole_tip_offset
    peg_insert_16mm: kpts.Offset = kpts.RECTANGULAR_HOLE_16MM.hole_tip_offset

    # Connector insert
    usba: kpts.Offset = kpts.USB_A_SOCKET.entry
    waterproof: kpts.Offset = kpts.WATERPROOF_SOCKET.entry
    bnc: kpts.Offset = kpts.BNC_SOCKET.plug_assembled
    dsub: kpts.Offset = kpts.D_SUB_SOCKET.entry
    rj45: kpts.Offset = kpts.RJ45_SOCKET.entry

    default: kpts.Offset = nut_thread_m16


def _dist(a: kpts.Offset, b: kpts.Offset) -> tuple[float, float, float]:
    """Position delta from offset *a* to offset *b* [m]."""
    return (b.pos[0] - a.pos[0], b.pos[1] - a.pos[1], b.pos[2] - a.pos[2])


@configclass
class FactoryAssemblyProfileCfg(PresetCfg):
    """Assembly profile per task variant.

    Each field is an :class:`AssemblyProfileCfg` describing the full assembly
    path geometry (segment endpoints, screw ratios, start noise).
    """

    # Nut threading — IncrementalSegment with screw pitch
    nut_thread_m4: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.BOLT_M4.fully_screwed_nut_offset,
                distance=_dist(kpts.BOLT_M4.fully_screwed_nut_offset, kpts.BOLT_M4.bolt_tip_offset),
                ratio=(0.0, 0.0, kpts.NUT_M4.screw_ratio / (2.0 * math.pi)),
            )
        ]
    )
    nut_thread_m8: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.BOLT_M8.fully_screwed_nut_offset,
                distance=_dist(kpts.BOLT_M8.fully_screwed_nut_offset, kpts.BOLT_M8.bolt_tip_offset),
                ratio=(0.0, 0.0, kpts.NUT_M8.screw_ratio / (2.0 * math.pi)),
            )
        ]
    )
    nut_thread_m12: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.BOLT_M12.fully_screwed_nut_offset,
                distance=_dist(kpts.BOLT_M12.fully_screwed_nut_offset, kpts.BOLT_M12.bolt_tip_offset),
                ratio=(0.0, 0.0, kpts.NUT_M12.screw_ratio / (2.0 * math.pi)),
            )
        ]
    )
    nut_thread_m16: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.BOLT_M16.fully_screwed_nut_offset,
                distance=_dist(kpts.BOLT_M16.fully_screwed_nut_offset, kpts.BOLT_M16.bolt_tip_offset),
                ratio=(0.0, 0.0, kpts.NUT_M16.screw_ratio / (2.0 * math.pi)),
            )
        ]
    )

    # Gear mesh — fraction starts at 0.1 so teeth are clear before yaw noise
    gear_mesh_small: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.1, 1.0),
                start_sampler=SymmetryOrbitCfg(symmetry=_SYM_CONTINUOUS),
                start_pose=kpts.GEAR_BASE.small_gear_assembled_bottom_offset,
                distance=_dist(kpts.GEAR_BASE.small_gear_assembled_bottom_offset, kpts.GEAR_BASE.small_gear_tip_offset),
            )
        ]
    )
    gear_mesh_medium: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.1, 1.0),
                start_sampler=SymmetryOrbitCfg(symmetry=_SYM_CONTINUOUS),
                start_pose=kpts.GEAR_BASE.medium_gear_assembled_bottom_offset,
                distance=_dist(
                    kpts.GEAR_BASE.medium_gear_assembled_bottom_offset, kpts.GEAR_BASE.medium_gear_tip_offset
                ),
            )
        ]
    )
    gear_mesh_large: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.1, 1.0),
                start_sampler=SymmetryOrbitCfg(symmetry=_SYM_CONTINUOUS),
                start_pose=kpts.GEAR_BASE.large_gear_assembled_bottom_offset,
                distance=_dist(kpts.GEAR_BASE.large_gear_assembled_bottom_offset, kpts.GEAR_BASE.large_gear_tip_offset),
            )
        ]
    )

    # Rod insert (round) — any yaw is valid
    rod_insert_4mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=SymmetryOrbitCfg(symmetry=_SYM_CONTINUOUS),
                start_pose=kpts.HOLE_4MM.inserted_peg_base_offset,
                distance=_dist(kpts.HOLE_4MM.inserted_peg_base_offset, kpts.HOLE_4MM.hole_tip_offset),
            )
        ]
    )
    rod_insert_8mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=SymmetryOrbitCfg(symmetry=_SYM_CONTINUOUS),
                start_pose=kpts.HOLE_8MM.inserted_peg_base_offset,
                distance=_dist(kpts.HOLE_8MM.inserted_peg_base_offset, kpts.HOLE_8MM.hole_tip_offset),
            )
        ]
    )
    rod_insert_12mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=SymmetryOrbitCfg(symmetry=_SYM_CONTINUOUS),
                start_pose=kpts.HOLE_12MM.inserted_peg_base_offset,
                distance=_dist(kpts.HOLE_12MM.inserted_peg_base_offset, kpts.HOLE_12MM.hole_tip_offset),
            )
        ]
    )
    rod_insert_16mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_sampler=SymmetryOrbitCfg(symmetry=_SYM_CONTINUOUS),
                start_pose=kpts.HOLE_16MM.inserted_peg_base_offset,
                distance=_dist(kpts.HOLE_16MM.inserted_peg_base_offset, kpts.HOLE_16MM.hole_tip_offset),
            )
        ]
    )

    # Peg insert (rectangular) — discrete yaw symmetry
    peg_insert_4mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.0, 0.7),
                start_sampler=SymmetryOrbitCfg(symmetry=_SYM_4FOLD),
                start_pose=kpts.RECTANGULAR_HOLE_4MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_4MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_4MM.hole_tip_offset
                ),
            ),
            IncrementalSegmentCfg(
                fraction=(0.7, 1.5),
                start_sampler=UniformPoseNoiseCfg(
                    x=(-0.01, 0.01), y=(-0.01, 0.01), roll=(-0.3, 0.3), pitch=(-0.3, 0.3), yaw=(-3.14, 3.14)
                ),
                start_pose=kpts.RECTANGULAR_HOLE_4MM.one_mm_above_hole_tip_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_4MM.one_mm_above_hole_tip_offset,
                    kpts.RECTANGULAR_HOLE_4MM.above_hole_tip_offset,
                ),
            ),
        ]
    )
    peg_insert_8mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.0, 0.7),
                start_sampler=SymmetryOrbitCfg(symmetry=_SYM_2FOLD),
                start_pose=kpts.RECTANGULAR_HOLE_8MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_8MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_8MM.hole_tip_offset
                ),
            ),
            IncrementalSegmentCfg(
                fraction=(0.7, 1.5),
                start_sampler=UniformPoseNoiseCfg(
                    x=(-0.01, 0.01), y=(-0.01, 0.01), roll=(-0.3, 0.3), pitch=(-0.3, 0.3), yaw=(-3.14, 3.14)
                ),
                start_pose=kpts.RECTANGULAR_HOLE_8MM.one_mm_above_hole_tip_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_8MM.one_mm_above_hole_tip_offset,
                    kpts.RECTANGULAR_HOLE_8MM.above_hole_tip_offset,
                ),
            ),
        ]
    )
    peg_insert_12mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.0, 0.7),
                start_sampler=SymmetryOrbitCfg(symmetry=_SYM_2FOLD),
                start_pose=kpts.RECTANGULAR_HOLE_12MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_12MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_12MM.hole_tip_offset
                ),
            ),
            IncrementalSegmentCfg(
                fraction=(0.7, 1.5),
                start_sampler=UniformPoseNoiseCfg(
                    x=(-0.01, 0.01), y=(-0.01, 0.01), roll=(-0.3, 0.3), pitch=(-0.3, 0.3), yaw=(-3.14, 3.14)
                ),
                start_pose=kpts.RECTANGULAR_HOLE_12MM.one_mm_above_hole_tip_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_12MM.one_mm_above_hole_tip_offset,
                    kpts.RECTANGULAR_HOLE_12MM.above_hole_tip_offset,
                ),
            ),
        ]
    )
    peg_insert_16mm: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                fraction=(0.0, 0.7),
                start_sampler=SymmetryOrbitCfg(symmetry=_SYM_2FOLD),
                start_pose=kpts.RECTANGULAR_HOLE_16MM.inserted_peg_base_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_16MM.inserted_peg_base_offset, kpts.RECTANGULAR_HOLE_16MM.hole_tip_offset
                ),
            ),
            IncrementalSegmentCfg(
                fraction=(0.7, 1.5),
                start_sampler=UniformPoseNoiseCfg(
                    x=(-0.01, 0.01), y=(-0.01, 0.01), roll=(-0.3, 0.3), pitch=(-0.3, 0.3), yaw=(-3.14, 3.14)
                ),
                start_pose=kpts.RECTANGULAR_HOLE_16MM.one_mm_above_hole_tip_offset,
                distance=_dist(
                    kpts.RECTANGULAR_HOLE_16MM.one_mm_above_hole_tip_offset,
                    kpts.RECTANGULAR_HOLE_16MM.above_hole_tip_offset,
                ),
            ),
        ]
    )

    # Connector insert — pure linear, no yaw noise (keyed)
    usba: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.USB_A_SOCKET.plug_assembled,
                distance=_dist(kpts.USB_A_SOCKET.plug_assembled, kpts.USB_A_SOCKET.entry),
            )
        ]
    )
    waterproof: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.WATERPROOF_SOCKET.plug_assembled,
                distance=_dist(kpts.WATERPROOF_SOCKET.plug_assembled, kpts.WATERPROOF_SOCKET.entry),
            )
        ]
    )
    dsub: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.D_SUB_SOCKET.plug_assembled,
                distance=_dist(kpts.D_SUB_SOCKET.plug_assembled, kpts.D_SUB_SOCKET.entry),
            )
        ]
    )
    rj45: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            IncrementalSegmentCfg(
                start_pose=kpts.RJ45_SOCKET.plug_assembled,
                distance=_dist(kpts.RJ45_SOCKET.plug_assembled, kpts.RJ45_SOCKET.entry),
            )
        ]
    )

    # BNC — two-segment: linear insertion then 90-deg bayonet twist
    bnc: AssemblyProfileCfg = AssemblyProfileCfg(
        segments=[
            EndPointsSegmentCfg(
                fraction=(0.0, 0.4),
                start_pose=kpts.BNC_SOCKET.plug_assembled,
                end_pose=kpts.BNC_SOCKET.insert_start,
                revolutions=(0.0, 0.0, 0.25),
            ),
            EndPointsSegmentCfg(
                fraction=(0.4, 1.0),
                start_pose=kpts.BNC_SOCKET.insert_start,
                end_pose=kpts.BNC_SOCKET.entry,
            ),
        ]
    )

    default: AssemblyProfileCfg = nut_thread_m16


@configclass
class HeldAssetAlignOffsetCfg(PresetCfg):
    # Nut threading — bottom of nut center axis for alignment
    nut_thread_m4: kpts.Offset = kpts.NUT_M4.center_axis_bottom
    nut_thread_m8: kpts.Offset = kpts.NUT_M8.center_axis_bottom
    nut_thread_m12: kpts.Offset = kpts.NUT_M12.center_axis_bottom
    nut_thread_m16: kpts.Offset = kpts.NUT_M16.center_axis_bottom

    # Gear mesh
    gear_mesh_small: kpts.Offset = kpts.SMALL_GEAR.center_axis_bottom
    gear_mesh_medium: kpts.Offset = kpts.MEDIUM_GEAR.center_axis_bottom
    gear_mesh_large: kpts.Offset = kpts.LARGE_GEAR.center_axis_bottom

    # Rod insert (round)
    rod_insert_4mm: kpts.Offset = kpts.ROD_4MM.center_axis_bottom
    rod_insert_8mm: kpts.Offset = kpts.ROD_8MM.center_axis_bottom
    rod_insert_12mm: kpts.Offset = kpts.ROD_12MM.center_axis_bottom
    rod_insert_16mm: kpts.Offset = kpts.ROD_16MM.center_axis_bottom

    # Peg insert (rectangular) — peg tip is the alignment reference
    peg_insert_4mm: kpts.Offset = kpts.RECTANGULAR_PEG_4MM.peg_tip
    peg_insert_8mm: kpts.Offset = kpts.RECTANGULAR_PEG_8MM.peg_tip
    peg_insert_12mm: kpts.Offset = kpts.RECTANGULAR_PEG_12MM.peg_tip
    peg_insert_16mm: kpts.Offset = kpts.RECTANGULAR_PEG_16MM.peg_tip

    # Connector insert — insertion tip of the plug
    usba: kpts.Offset = kpts.USB_A_PLUG.insertion_tip
    waterproof: kpts.Offset = kpts.WATERPROOF_PLUG.insertion_tip
    bnc: kpts.Offset = kpts.BNC_PLUG.insertion_tip
    dsub: kpts.Offset = kpts.D_SUB_PLUG.insertion_tip
    rj45: kpts.Offset = kpts.RJ45_PLUG.insertion_tip

    default: kpts.Offset = nut_thread_m16


@configclass
class HeldAssetSymmetryCfg(PresetCfg):
    """Held-asset symmetry per task variant, consumed by the
    :class:`~...utils.symmetry.Symmetry` and the assembly sampler. Each value
    is an :class:`~...utils.symmetry.AssetSymmetryCfg`; the shared ``_SYM_*``
    constants below are the single source of truth (continuous / N-fold / none)."""

    # Threaded nuts / round rods / gears: continuous yaw symmetry (yaw is free).
    nut_thread_m4: AssetSymmetryCfg = _SYM_CONTINUOUS
    nut_thread_m8: AssetSymmetryCfg = _SYM_CONTINUOUS
    nut_thread_m12: AssetSymmetryCfg = _SYM_CONTINUOUS
    nut_thread_m16: AssetSymmetryCfg = _SYM_CONTINUOUS
    rod_insert_4mm: AssetSymmetryCfg = _SYM_CONTINUOUS
    rod_insert_8mm: AssetSymmetryCfg = _SYM_CONTINUOUS
    rod_insert_12mm: AssetSymmetryCfg = _SYM_CONTINUOUS
    rod_insert_16mm: AssetSymmetryCfg = _SYM_CONTINUOUS
    gear_mesh_small: AssetSymmetryCfg = _SYM_CONTINUOUS
    gear_mesh_medium: AssetSymmetryCfg = _SYM_CONTINUOUS
    gear_mesh_large: AssetSymmetryCfg = _SYM_CONTINUOUS

    # Pegs: discrete yaw symmetry (square = 4-fold, rectangular = 2-fold), matching
    # the DiscreteYaw orders in the assembly profiles.
    peg_insert_4mm: AssetSymmetryCfg = _SYM_4FOLD
    peg_insert_8mm: AssetSymmetryCfg = _SYM_2FOLD
    peg_insert_12mm: AssetSymmetryCfg = _SYM_2FOLD
    peg_insert_16mm: AssetSymmetryCfg = _SYM_2FOLD

    # Keyed connectors: no rotational symmetry.
    usba: AssetSymmetryCfg = _SYM_NONE
    waterproof: AssetSymmetryCfg = _SYM_NONE
    bnc: AssetSymmetryCfg = _SYM_NONE
    dsub: AssetSymmetryCfg = _SYM_NONE
    rj45: AssetSymmetryCfg = _SYM_NONE

    default: AssetSymmetryCfg = _SYM_NONE


_GEAR_GRASPED_RANGE = dict(
    x=(-0.02, 0.02),
    y=(-0.02, 0.02),
    z=(0.035, 0.045),
    roll=(3.141, 3.141),
    pitch=(-0.5, 0.5),
    yaw=(-2.09, 2.09),
)
_INSERT_GRASPED_RANGE = dict(
    x=(-0.005, 0.005),
    y=(-0.005, 0.005),
    z=(0.047, 0.057),
    roll=(3.141, 3.141),
    pitch=(-0.5, 0.5),
    yaw=(-2.09, 2.09),
)
