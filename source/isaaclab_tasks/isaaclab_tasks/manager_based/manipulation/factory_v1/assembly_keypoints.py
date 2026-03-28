# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keypoint definitions for all NIST Factory assembly assets.

All quaternions use IsaacLab's (x, y, z, w) convention.
Positions are in meters relative to the asset's root frame.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject


@configclass
class Offset:
    """A position + quaternion offset relative to an asset root frame.

    Args:
        pos: Translation offset [m].
        quat: Orientation offset in (x, y, z, w).
    """

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    @property
    def pose(self) -> tuple[float, float, float, float, float, float, float]:
        return self.pos + self.quat

    def apply(self, root: RigidObject | Articulation) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform this offset into world frame using the asset's current root pose.

        Args:
            root: The asset whose root pose defines the parent frame.

        Returns:
            A tuple of (position, quaternion) tensors in world frame,
            each with shape (num_envs, 3) and (num_envs, 4) respectively.
        """
        root_pos = wp.to_torch(root.data.root_pos_w)
        root_quat = wp.to_torch(root.data.root_quat_w)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            root_pos,
            root_quat,
            torch.tensor(self.pos, device=root_pos.device).repeat(root_pos.shape[0], 1),
            torch.tensor(self.quat, device=root_pos.device).repeat(root_pos.shape[0], 1),
        )
        return pos_w, quat_w

    def combine(self, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose this offset on top of an arbitrary parent frame.

        Unlike :meth:`apply`, the parent frame is given directly as tensors
        rather than read from an asset.

        Args:
            pos: Parent frame positions [m], shape (N, 3).
            quat: Parent frame quaternions (x, y, z, w), shape (N, 4).

        Returns:
            A tuple of (position, quaternion) tensors for the composed frame.
        """
        num_data = pos.shape[0]
        device = pos.device
        return math_utils.combine_frame_transforms(
            pos,
            quat,
            torch.tensor(self.pos, device=device).repeat(num_data, 1),
            torch.tensor(self.quat, device=device).repeat(num_data, 1),
        )


# =============================================================================
# Kit Tray — positions of assets sitting in the tray
# =============================================================================


@configclass
class KitTrayKeyPointsCfg:
    """Offsets of each asset's resting pose in the kit tray, relative to the tray root."""

    kit_tray_center: Offset = Offset(pos=(0.0, 0.0, 0.0))
    bnc_plug: Offset = Offset(pos=(0.0954, 0.0635, -0.0108), quat=(0.0, 0.0, 0.7071, 0.70711))
    dsub_plug: Offset = Offset(pos=(0.0156, 0.0327, 0.006), quat=(0.5, -0.5, -0.5, 0.5))
    rj45_plug: Offset = Offset(pos=(-0.15, 0.0821, 0.006), quat=(0.5, 0.5, -0.5, 0.5))
    waterproof_plug: Offset = Offset(pos=(0.1345, -0.1051, -0.015), quat=(0.0, 0.0, 0.7071, 0.70711))
    usba_plug: Offset = Offset(pos=(-0.1834, -0.0906, 0.006), quat=(0.0, 0.7071, 0.0, 0.7071))

    nut_m4: Offset = Offset(pos=(0.1169, 0.1427, -0.0049), quat=(0.0, 0.0, 0.9659, 0.2588))
    nut_m8: Offset = Offset(pos=(0.0646, -0.0052, -0.0091), quat=(0.0, 0.0, -0.96593, 0.25882))
    nut_m12: Offset = Offset(pos=(-0.0112, -0.1362, -0.013), quat=(0.0, 0.0, -0.96593, 0.25882))
    nut_m16: Offset = Offset(pos=(-0.1078, -0.0177, 0.0118), quat=(0.68301, 0.18301, -0.18301, 0.68301))

    rectangular_peg_4mm: Offset = Offset(pos=(-0.1482, -0.1419, 0.006), quat=(0.0, 0.7071, 0.0, 0.7071))
    rectangular_peg_8mm: Offset = Offset(pos=(-0.0899, 0.0836, 0.006), quat=(0.5, 0.5, -0.5, 0.5))
    rectangular_peg_12mm: Offset = Offset(pos=(-0.0248, 0.135, 0.006), quat=(0.5, 0.5, -0.5, -0.5))
    rectangular_peg_16mm: Offset = Offset(pos=(0.1432, 0.0886, 0.005), quat=(0.0, 0.0, 0.7071, 0.7071))

    rod_4mm: Offset = Offset(pos=(0.1399, 0.0297, 0.01), quat=(0.7071, 0.0, 0.0, 0.70711))
    rod_8mm: Offset = Offset(pos=(-0.1408, 0.1306, 0.005), quat=(0.0, 0.0, 0.0, 1.0))
    rod_12mm: Offset = Offset(pos=(-0.0592, 0.005, 0.005), quat=(0.0, 0.0, 0.7071, -0.70711))
    rod_16mm: Offset = Offset(pos=(0.0704, -0.074, 0.005), quat=(0.0, 0.0, 0.7071, -0.70711))

    large_gear: Offset = Offset(pos=(0.039, 0.086, 0.005), quat=(0.0, 0.0, 0.7071, -0.70711))
    medium_gear: Offset = Offset(pos=(0.0061, -0.0475, 0.005), quat=(0.0, 0.0, -0.70711, 0.7071))
    small_gear: Offset = Offset(pos=(0.0595, -0.0875, 0.005), quat=(0.0, 0.0, 0.7071, -0.70711))


# =============================================================================
# NIST Board — socket/hole positions on the board
# =============================================================================


@configclass
class NistBoardKeyPointsCfg:
    """Target placement offsets for each asset on the NIST task board, relative to the board root.

    Used by reset functions to position assets at their correct board locations.
    Each field name matches the ``asset_map`` keys in :class:`FixedAssetMapCfg`.
    """

    nist_board_center: Offset = Offset(pos=(0.197176, -0.19145, 0.0))

    bnc_plug: Offset = Offset(pos=(0.2797, -0.1915, 0.0), quat=(0.7071, 0.7071, 0.0, 0.0))
    bnc_socket: Offset = Offset(pos=(0.2797, -0.1915, 0.0), quat=(0.7071, 0.7071, 0.0, 0.0))
    dsub_plug: Offset = Offset(pos=(0.2129, -0.2659, -0.019), quat=(0.7071, -0.7071, 0.0, 0.0))
    dsub_socket: Offset = Offset(pos=(0.2129, -0.2659, -0.019), quat=(0.7071, -0.7071, 0.0, 0.0))
    rj45_plug: Offset = Offset(pos=(0.3473, -0.3415, 0.0), quat=(0.0, 1.0, 0.0, 0.0))
    rj45_socket: Offset = Offset(pos=(0.3473, -0.3415, 0.0), quat=(0.0, 1.0, 0.0, 0.0))
    waterproof_plug: Offset = Offset(pos=(0.1981, -0.1166, -0.0002), quat=(0.0, 1.0, 0.0, 0.0))
    waterproof_socket: Offset = Offset(pos=(0.1981, -0.1166, -0.0002), quat=(0.0, 1.0, 0.0, 0.0))
    usba_plug: Offset = Offset(pos=(0.2721, -0.0415, -0.0001), quat=(0.7071, 0.7071, 0.0, 0.0))
    usba_socket: Offset = Offset(pos=(0.2721, -0.0415, -0.0001), quat=(0.7071, 0.7071, 0.0, 0.0))

    nut_m4: Offset = Offset(pos=(0.1223, -0.1914, 0.004), quat=(1.0, 0.0, 0.0, 0.0))
    bolt_m4: Offset = Offset(pos=(0.1223, -0.1914, 0.0131), quat=(1.0, 0.0, 0.0, 0.0))
    nut_m8: Offset = Offset(pos=(0.0473, -0.0407, 0.0085), quat=(1.0, 0.0, 0.0, 0.0))
    bolt_m8: Offset = Offset(pos=(0.0473, -0.0407, 0.0172), quat=(1.0, 0.0, 0.0, 0.0))
    nut_m12: Offset = Offset(pos=(0.3473, -0.2665, 0.0123), quat=(1.0, 0.0, 0.0, 0.0))
    bolt_m12: Offset = Offset(pos=(0.3473, -0.2665, 0.0212), quat=(1.0, 0.0, 0.0, 0.0))
    nut_m16: Offset = Offset(pos=(0.04715, -0.3416, 0.0094), quat=(1.0, 0.0, 0.0, 0.0))
    bolt_m16: Offset = Offset(pos=(0.04715, -0.3416, 0.0194), quat=(1.0, 0.0, 0.0, 0.0))

    rectangular_peg_4mm: Offset = Offset(pos=(0.1971, -0.1915, -0.0003), quat=(0.7071, -0.7071, 0.0, 0.0))
    rectangular_hole_4mm: Offset = Offset(pos=(0.1971, -0.1915, -0.0003), quat=(0.7071, -0.7071, 0.0, 0.0))
    rectangular_peg_8mm: Offset = Offset(pos=(0.2717, -0.2659, -0.0003), quat=(0.7071, -0.7071, 0.0, 0.0))
    rectangular_hole_8mm: Offset = Offset(pos=(0.2717, -0.2659, -0.0003), quat=(0.7071, -0.7071, 0.0, 0.0))
    rectangular_peg_12mm: Offset = Offset(pos=(0.1971, -0.0413, -0.0003), quat=(0.0, 1.0, 0.0, 0.0))
    rectangular_hole_12mm: Offset = Offset(pos=(0.1971, -0.0413, -0.0003), quat=(0.0, 1.0, 0.0, 0.0))
    rectangular_peg_16mm: Offset = Offset(pos=(0.3472, -0.0413, -0.0003), quat=(0.7071, 0.7071, 0.0, 0.0))
    rectangular_hole_16mm: Offset = Offset(pos=(0.3472, -0.0413, -0.0003), quat=(0.7071, 0.7071, 0.0, 0.0))

    rod_4mm: Offset = Offset(pos=(0.3473, -0.1918, -0.0001), quat=(0.7071, -0.7071, 0.0, 0.0))
    hole_4mm: Offset = Offset(pos=(0.3473, -0.1918, -0.0001), quat=(0.7071, -0.7071, 0.0, 0.0))
    rod_8mm: Offset = Offset(pos=(0.3473, -0.1164, -0.0001), quat=(0.7071, -0.7071, 0.0, 0.0))
    hole_8mm: Offset = Offset(pos=(0.3473, -0.1164, -0.0001), quat=(0.7071, -0.7071, 0.0, 0.0))
    rod_12mm: Offset = Offset(pos=(0.1226, -0.0422, -0.0001), quat=(0.7071, -0.7071, 0.0, 0.0))
    hole_12mm: Offset = Offset(pos=(0.1226, -0.0422, -0.0001), quat=(0.7071, -0.7071, 0.0, 0.0))
    rod_16mm: Offset = Offset(pos=(0.1221, -0.2665, -0.0001), quat=(0.7071, -0.7071, 0.0, 0.0))
    hole_16mm: Offset = Offset(pos=(0.1221, -0.2665, -0.0001), quat=(0.7071, -0.7071, 0.0, 0.0))

    large_gear: Offset = Offset(pos=(0.0474, -0.1713, -0.0002), quat=(0.7071, -0.7071, 0.0, 0.0))
    medium_gear: Offset = Offset(pos=(0.0459, -0.1714, -0.0002), quat=(0.73566, -0.67736, 0.0, 0.0))
    small_gear: Offset = Offset(pos=(0.0474, -0.1713, -0.0002), quat=(0.7071, -0.7071, 0.0, 0.0))
    gear_base: Offset = Offset(pos=(0.0474, -0.1713, -0.0002), quat=(0.7071, -0.7071, 0.0, 0.0))


# =============================================================================
# Bolt keypoints (all sizes)
# =============================================================================


@configclass
class BoltM16KeyPointsCfg:
    """Keypoints along the M16 bolt shaft, from head to tip.

    Thread offsets are measured from the bolt head (z=0) upward along the shaft axis.
    """

    one_cm_above_tip: Offset = Offset(pos=(0.0, 0.0, 0.045))
    bolt_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.035))
    first_thread: Offset = Offset(pos=(0.0, 0.0, 0.034))
    second_thread: Offset = Offset(pos=(0.0, 0.0, 0.032))
    third_thread: Offset = Offset(pos=(0.0, 0.0, 0.03))
    fully_screwed_nut_offset: Offset = Offset(pos=(0.0, 0.0, 0.022))
    eighth_thread_nist_thread: Offset = Offset(pos=(0.0, 0.0, 0.02))
    full_thread: Offset = Offset(pos=(0.0, 0.0, 0.01))


@configclass
class BoltM12KeyPointsCfg:
    bolt_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.035))
    second_thread: Offset = Offset(pos=(0.0, 0.0, 0.0285))
    fully_screwed_nut_offset: Offset = Offset(pos=(0.0, 0.0, 0.022))
    seventh_thread: Offset = Offset(pos=(0.0, 0.0, 0.0215))
    head: Offset = Offset(pos=(0.0, 0.0, 0.0))


@configclass
class BoltM8KeyPointsCfg:
    bolt_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.026))
    second_thread: Offset = Offset(pos=(0.0, 0.0, 0.0242))
    seventh_thread: Offset = Offset(pos=(0.0, 0.0, 0.0182))
    fully_screwed_nut_offset: Offset = Offset(pos=(0.0, 0.0, 0.018))
    full_thread: Offset = Offset(pos=(0.0, 0.0, 0.0084))
    head: Offset = Offset(pos=(0.0, 0.0, 0.0))


@configclass
class BoltM4KeyPointsCfg:
    bolt_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.02))
    second_thread: Offset = Offset(pos=(0.0, 0.0, 0.0189))
    fully_screwed_nut_offset: Offset = Offset(pos=(0.0, 0.0, 0.014))
    tenth_thread: Offset = Offset(pos=(0.0, 0.0, 0.0134))
    full_thread: Offset = Offset(pos=(0.0, 0.0, 0.0044))
    head: Offset = Offset(pos=(0.0, 0.0, 0.0))


# =============================================================================
# Nut keypoints (all sizes)
# =============================================================================


@configclass
class NutM16KeyPointsCfg:
    """Keypoints for the M16 nut.

    ``grasp_point`` includes a 90-degree rotation around z so the gripper approaches
    from the side. ``screw_ratio`` [m/rad] converts rotation to linear displacement
    along the bolt axis.
    """

    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.01), quat=(0.0, 0.0, -0.7071, 0.7071))
    grasp_diameter: float = 0.024
    center_axis_bottom: Offset = Offset(pos=(0.0, 0.0, 0.01))
    center_axis_middle: Offset = Offset(pos=(0.0, 0.0, 0.0165))
    center_axis_top: Offset = Offset(pos=(0.0, 0.0, 0.023))
    screw_ratio: float = 0.002


@configclass
class NutM12KeyPointsCfg:
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.013), quat=(0.0, 0.0, -0.7071, 0.7071))
    grasp_diameter: float = 0.019
    center_axis_bottom: Offset = Offset(pos=(0.0, 0.0, 0.013))
    center_axis_middle: Offset = Offset(pos=(0.0, 0.0, 0.018))
    center_axis_top: Offset = Offset(pos=(0.0, 0.0, 0.023))
    screw_ratio: float = 0.00175


@configclass
class NutM8KeyPointsCfg:
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.0093), quat=(0.0, 0.0, -0.7071, 0.7071))
    grasp_diameter: float = 0.013
    center_axis_bottom: Offset = Offset(pos=(0.0, 0.0, 0.0093))
    center_axis_middle: Offset = Offset(pos=(0.0, 0.0, 0.0126))
    center_axis_top: Offset = Offset(pos=(0.0, 0.0, 0.016))
    screw_ratio: float = 0.00125


@configclass
class NutM4KeyPointsCfg:
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.0046), quat=(0.0, 0.0, -0.7071, 0.7071))
    grasp_diameter: float = 0.007
    center_axis_bottom: Offset = Offset(pos=(0.0, 0.0, 0.0046))
    center_axis_middle: Offset = Offset(pos=(0.0, 0.0, 0.0064))
    center_axis_top: Offset = Offset(pos=(0.0, 0.0, 0.0082))
    screw_ratio: float = 0.0007


# =============================================================================
# Gear keypoints
# =============================================================================


@configclass
class GearBaseKeyPointsCfg:
    """Keypoints for the three gear shafts on the gear base fixture.

    Each shaft has a tip (top) and bottom offset. The x-coordinate distinguishes
    the three shafts (small, medium, large) on the base.
    """

    small_gear_tip_offset: Offset = Offset(pos=(0.0508, 0.0, 0.025))
    small_gear_assembled_bottom_offset: Offset = Offset(pos=(0.05075, 0.0, 0.005))
    medium_gear_tip_offset: Offset = Offset(pos=(0.02025, 0.0, 0.025))
    medium_gear_assembled_bottom_offset: Offset = Offset(pos=(0.02025, 0.0, 0.005))
    large_gear_tip_offset: Offset = Offset(pos=(-0.0303, 0.0, 0.025))
    large_gear_assembled_bottom_offset: Offset = Offset(pos=(-0.0303, 0.0, 0.005))


@configclass
class SmallGearKeyPointsCfg:
    center_axis_bottom: Offset = Offset(pos=(0.05075, 0.0, 0.005))
    center_axis_top: Offset = Offset(pos=(0.05075, 0.0, 0.03))
    grasp_point: Offset = Offset(pos=(0.05075, 0.0, 0.022))
    grasp_diameter: float = 0.0175


@configclass
class MediumGearKeyPointsCfg:
    center_axis_bottom: Offset = Offset(pos=(0.02025, 0.0, 0.005))
    center_axis_top: Offset = Offset(pos=(0.02025, 0.0, 0.03))
    grasp_point: Offset = Offset(pos=(0.02025, 0.0, 0.022))
    grasp_diameter: float = 0.03


@configclass
class LargeGearKeyPointsCfg:
    center_axis_bottom: Offset = Offset(pos=(-0.0303, 0.0, 0.005))
    center_axis_top: Offset = Offset(pos=(-0.0303, 0.0, 0.03))
    grasp_point: Offset = Offset(pos=(-0.0303, 0.0, 0.022))
    grasp_diameter: float = 0.03


# =============================================================================
# Round hole / rod keypoints (all sizes)
# =============================================================================


@configclass
class Hole16MMKeyPointsCfg:
    hole_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.009))
    inserted_peg_base_offset: Offset = Offset(pos=(0.0, 0.0, 0.0))


@configclass
class Rod16MMKeyPointsCfg:
    geometry_origin: Offset = Offset(pos=(0.0, 0.0, 0.025))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.035))
    center_axis_bottom: Offset = Offset(pos=(0.0, 0.0, 0.0))
    grasp_diameter: float = 0.016


@configclass
class Hole12MMKeyPointsCfg:
    hole_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.009))
    inserted_peg_base_offset: Offset = Offset(pos=(0.0, 0.0, 0.0))


@configclass
class Rod12MMKeyPointsCfg:
    geometry_origin: Offset = Offset(pos=(0.0, 0.0, 0.025))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.035))
    center_axis_bottom: Offset = Offset(pos=(0.0, 0.0, 0.0))
    grasp_diameter: float = 0.012


@configclass
class Hole8MMKeyPointsCfg:
    hole_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.009))
    inserted_peg_base_offset: Offset = Offset(pos=(0.0, 0.0, 0.0))


@configclass
class Rod8MMKeyPointsCfg:
    geometry_origin: Offset = Offset(pos=(0.0, 0.0, 0.025))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.035))
    center_axis_bottom: Offset = Offset(pos=(0.0, 0.0, 0.0))
    grasp_diameter: float = 0.008


@configclass
class Hole4MMKeyPointsCfg:
    hole_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.009))
    inserted_peg_base_offset: Offset = Offset(pos=(0.0, 0.0, 0.0))


@configclass
class Rod4MMKeyPointsCfg:
    geometry_origin: Offset = Offset(pos=(0.0, 0.0, 0.025))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.035))
    center_axis_bottom: Offset = Offset(pos=(0.0, 0.0, 0.0))
    grasp_diameter: float = 0.004


# =============================================================================
# Rectangular peg / hole keypoints (all sizes)
# =============================================================================


@configclass
class RectangularPeg16MMKeyPointsCfg:
    geometry_origin: Offset = Offset(pos=(0.0, 0.0, 0.025))
    peg_tip: Offset = Offset(pos=(0.0, 0.0, 0.0))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.035))
    grasp_diameter: float = 0.01


@configclass
class RectangularHole16MMKeyPointsCfg:
    hole_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.009))
    inserted_peg_base_offset: Offset = Offset(pos=(0.0, 0.0, 0.0))


@configclass
class RectangularPeg12MMKeyPointsCfg:
    geometry_origin: Offset = Offset(pos=(0.0, 0.0, 0.025))
    peg_tip: Offset = Offset(pos=(0.0, 0.0, 0.0))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.035))
    grasp_diameter: float = 0.008


@configclass
class RectangularHole12MMKeyPointsCfg:
    hole_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.009))
    inserted_peg_base_offset: Offset = Offset(pos=(0.0, 0.0, 0.0))


@configclass
class RectangularPeg8MMKeyPointsCfg:
    geometry_origin: Offset = Offset(pos=(0.0, 0.0, 0.025))
    peg_tip: Offset = Offset(pos=(0.0, 0.0, 0.0))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.035))
    grasp_diameter: float = 0.008


@configclass
class RectangularHole8MMKeyPointsCfg:
    hole_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.009))
    inserted_peg_base_offset: Offset = Offset(pos=(0.0, 0.0, 0.0))


@configclass
class RectangularPeg4MMKeyPointsCfg:
    geometry_origin: Offset = Offset(pos=(0.0, 0.0, 0.025))
    peg_tip: Offset = Offset(pos=(0.0, 0.0, 0.0))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.035))
    grasp_diameter: float = 0.004


@configclass
class RectangularHole4MMKeyPointsCfg:
    hole_tip_offset: Offset = Offset(pos=(0.0, 0.0, 0.009))
    inserted_peg_base_offset: Offset = Offset(pos=(0.0, 0.0, 0.0))


# =============================================================================
# Connector keypoints
# =============================================================================


@configclass
class USBAPlugKeyPointsCfg:
    insertion_tip: Offset = Offset(pos=(0.0, 0.0, 0.0335))
    tail: Offset = Offset(pos=(0.0, 0.0, 0.093))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.0563))
    grasp_diameter: float = 0.0152


@configclass
class USBASocketKeyPointsCfg:
    entry: Offset = Offset(pos=(0.0, 0.0, 0.0416))
    plug_assembled: Offset = Offset(pos=(0.0, 0.0, 0.0335))
    housing_bottom: Offset = Offset(pos=(0.0, 0.0, 0.0))


@configclass
class WaterproofPlugKeyPointsCfg:
    insertion_tip: Offset = Offset(pos=(0.0, 0.0, 0.021))
    tail: Offset = Offset(pos=(0.0, 0.0, 0.0589))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.044))
    grasp_diameter: float = 0.03152


@configclass
class WaterproofSocketKeyPointsCfg:
    entry: Offset = Offset(pos=(0.0, 0.0, 0.034))
    plug_assembled: Offset = Offset(pos=(0.0, 0.0, 0.021))


@configclass
class BNCPlugKeyPointsCfg:
    insertion_tip: Offset = Offset(pos=(0.0, 0.0, 0.0212))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.0197))
    plug_entry: Offset = Offset(pos=(0.0, 0.0, 0.0109), quat=(0.0, 0.0, 0.7071, 0.7071))
    grasp_diameter: float = 0.0143


@configclass
class BNCSocketKeyPointsCfg:
    plug_assembled: Offset = Offset(pos=(0.0, 0.0, 0.0212))


@configclass
class DSUBPlugKeyPointsCfg:
    insertion_tip: Offset = Offset(pos=(0.0, 0.0, 0.0))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.0125))
    grasp_diameter: float = 0.013


@configclass
class DSUBSocketKeyPointsCfg:
    plug_assembled: Offset = Offset(pos=(0.0, 0.0, 0.0))
    entry: Offset = Offset(pos=(0.0, 0.0, 0.0061))


@configclass
class RJ45PlugKeyPointsCfg:
    insertion_tip: Offset = Offset(pos=(0.0, 0.0, 0.015))
    tail: Offset = Offset(pos=(0.0, 0.0, 0.0771))
    grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.0388))
    grasp_diameter: float = 0.011


@configclass
class RJ45SocketKeyPointsCfg:
    entry: Offset = Offset(pos=(0.0, 0.0, 0.028))
    plug_assembled: Offset = Offset(pos=(0.0, 0.0, 0.015))


# =============================================================================
# Robot keypoints
# =============================================================================


@configclass
class PandaHandKeyPointsCfg:
    """Grasp keypoints on the Franka Panda hand, relative to the ``panda_hand`` link.

    The 180-degree rotation around z flips the gripper so it faces downward
    for top-down grasps.
    """

    gripper_center_grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.107), quat=(0.0, 0.0, 1.0, 0.0))
    gripper_tip_grasp_point: Offset = Offset(pos=(0.0, 0.0, 0.112), quat=(0.0, 0.0, 1.0, 0.0))


@configclass
class RobotRootKeyPointsCfg:
    base: Offset = Offset(pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 1.0, 0.0))


# =============================================================================
# Module-level instances
# =============================================================================

NIST_BOARD_KEY_POINTS_CFG = NistBoardKeyPointsCfg()
KIT_TRAY_KEY_POINTS_CFG = KitTrayKeyPointsCfg()

BOLT_M16_KEY_POINTS = BoltM16KeyPointsCfg()
NUT_M16_KEY_POINTS = NutM16KeyPointsCfg()
BOLT_M12_KEY_POINTS = BoltM12KeyPointsCfg()
NUT_M12_KEY_POINTS = NutM12KeyPointsCfg()
BOLT_M8_KEY_POINTS = BoltM8KeyPointsCfg()
NUT_M8_KEY_POINTS = NutM8KeyPointsCfg()
BOLT_M4_KEY_POINTS = BoltM4KeyPointsCfg()
NUT_M4_KEY_POINTS = NutM4KeyPointsCfg()

KEY_POINTS_RECTANGULAR_PEG_16MM = RectangularPeg16MMKeyPointsCfg()
KEY_POINTS_RECTANGULAR_HOLE_16MM = RectangularHole16MMKeyPointsCfg()
KEY_POINTS_RECTANGULAR_PEG_12MM = RectangularPeg12MMKeyPointsCfg()
KEY_POINTS_RECTANGULAR_HOLE_12MM = RectangularHole12MMKeyPointsCfg()
KEY_POINTS_RECTANGULAR_PEG_8MM = RectangularPeg8MMKeyPointsCfg()
KEY_POINTS_RECTANGULAR_HOLE_8MM = RectangularHole8MMKeyPointsCfg()
KEY_POINTS_RECTANGULAR_PEG_4MM = RectangularPeg4MMKeyPointsCfg()
KEY_POINTS_RECTANGULAR_HOLE_4MM = RectangularHole4MMKeyPointsCfg()

KEY_POINTS_HOLE_16MM = Hole16MMKeyPointsCfg()
KEY_POINTS_ROD_16MM = Rod16MMKeyPointsCfg()
KEY_POINTS_HOLE_12MM = Hole12MMKeyPointsCfg()
KEY_POINTS_ROD_12MM = Rod12MMKeyPointsCfg()
KEY_POINTS_HOLE_8MM = Hole8MMKeyPointsCfg()
KEY_POINTS_ROD_8MM = Rod8MMKeyPointsCfg()
KEY_POINTS_HOLE_4MM = Hole4MMKeyPointsCfg()
KEY_POINTS_ROD_4MM = Rod4MMKeyPointsCfg()

KEY_POINTS_GEAR_BASE = GearBaseKeyPointsCfg()
KEY_POINTS_SMALL_GEAR = SmallGearKeyPointsCfg()
KEY_POINTS_MEDIUM_GEAR = MediumGearKeyPointsCfg()
KEY_POINTS_LARGE_GEAR = LargeGearKeyPointsCfg()

KEY_POINTS_USB_A_SOCKET = USBASocketKeyPointsCfg()
KEY_POINTS_USB_A_PLUG = USBAPlugKeyPointsCfg()
KEY_POINTS_WATERPROOF_PLUG = WaterproofPlugKeyPointsCfg()
KEY_POINTS_WATERPROOF_SOCKET = WaterproofSocketKeyPointsCfg()
KEY_POINTS_D_SUB_PLUG = DSUBPlugKeyPointsCfg()
KEY_POINTS_D_SUB_SOCKET = DSUBSocketKeyPointsCfg()
KEY_POINTS_BNC_PLUG = BNCPlugKeyPointsCfg()
KEY_POINTS_BNC_SOCKET = BNCSocketKeyPointsCfg()
KEY_POINTS_RJ45_PLUG = RJ45PlugKeyPointsCfg()
KEY_POINTS_RJ45_SOCKET = RJ45SocketKeyPointsCfg()

KEY_POINTS_PANDA_HAND = PandaHandKeyPointsCfg()
KEY_POINTS_ROBOT = RobotRootKeyPointsCfg()
