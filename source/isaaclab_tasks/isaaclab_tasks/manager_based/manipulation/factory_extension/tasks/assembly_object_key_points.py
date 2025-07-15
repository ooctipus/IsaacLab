from isaaclab.utils import configclass
from .assembly_key_points_cfg import ObjectKeyPointsCfg, KeyPointCfg, Offset, SymmetryOffsets


@configclass
class KitTrayKeyPointsCfg(ObjectKeyPointsCfg):
    kit_tray_center = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])
    bnc_plug = KeyPointCfg([Offset(pos=(0.0954, 0.0635, -0.0108), quat=(0.70711, 0.0000, 0.0000, 0.7071))])
    dsub_plug = KeyPointCfg([Offset(pos=(0.0156, 0.0327, 0.0060), quat=(0.5000, 0.5000, -0.5000, -0.5000))])
    rj_45_plug = KeyPointCfg([Offset(pos=(-0.1500, 0.0821, 0.0060), quat=(0.5000, 0.5000, 0.5000, -0.5000))])
    waterproof_plug = KeyPointCfg([Offset(pos=(0.1345, -0.1051, -0.015), quat=(0.70711, 0.0000, 0.0000, 0.7071))])
    usba_plug = KeyPointCfg([Offset(pos=(-0.1834, -0.0906, 0.0060), quat=(0.7071, 0.0000, 0.7071, 0.0000))])

    nut_m4 = KeyPointCfg([Offset(pos=(0.1169, 0.1427, -0.0049), quat=(0.2588, 0.0000, 0.0000, 0.9659))])
    nut_m8 = KeyPointCfg([Offset(pos=(0.0646, -0.0052, -0.0091), quat=(0.25882, 0.0000, 0.0000, -0.96593))])
    nut_m12 = KeyPointCfg([Offset(pos=(-0.0112, -0.1362, -0.0130), quat=(0.25882, 0.0000, 0.0000, -0.96593))])
    nut_m16 = KeyPointCfg([Offset(pos=(-0.1078, -0.0177, 0.0118), quat=(0.68301, 0.68301, 0.18301, -0.18301))])

    rectangular_peg_4mm = KeyPointCfg([Offset(pos=(-0.1482, -0.1419, 0.0060), quat=(0.7071, 0.0000, 0.7071, 0.0000))])
    rectangular_peg_8mm = KeyPointCfg([Offset(pos=(-0.0899, 0.0836, 0.0060), quat=(0.5000, 0.5000, 0.5000, -0.5000))])
    rectangular_peg_12mm = KeyPointCfg([Offset(pos=(-0.0248, 0.1350, 0.0060), quat=(-0.5000, 0.5000, 0.5000, -0.5000))])
    rectangular_peg_16mm = KeyPointCfg([Offset(pos=(0.1432, 0.0886, 0.005), quat=(0.7071, 0.0000, 0.0000, 0.7071))])

    rod_4mm = KeyPointCfg([Offset(pos=(0.1399, 0.0297, 0.010), quat=(0.70711, 0.7071, 0.0000, 0.0000))])
    rod_8mm = KeyPointCfg([Offset(pos=(-0.1408, 0.1306, 0.005), quat=(1.0000, 0.0000, 0.0000, 0.0000))])
    rod_12mm = KeyPointCfg([Offset(pos=(-0.0592, 0.0050, 0.005), quat=(-0.70711, 0.0000, 0.0000, 0.7071))])
    rod_16mm = KeyPointCfg([Offset(pos=(0.0704, -0.0740, 0.005), quat=(-0.70711, 0.0000, 0.0000, 0.7071))])

    large_gear = KeyPointCfg([Offset(pos=(0.0390, 0.0860, 0.0050), quat=(-0.70711, 0.0000, 0.0000, 0.7071))])
    medium_gear = KeyPointCfg([Offset(pos=(0.0061, -0.0475, 0.0050), quat=(0.7071, 0.0000, 0.0000, -0.70711))])
    small_gear = KeyPointCfg([Offset(pos=(0.0595, -0.0875, 0.0050), quat=(-0.70711, 0.0000, 0.0000, 0.7071))])


@configclass
class KitTrayKeyPointsCfgStanding(ObjectKeyPointsCfg):
    kit_tray_center = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])
    bnc_plug = KeyPointCfg([Offset(pos=(0.0954, 0.0635, -0.0108), quat=(0.70711, 0.0000, 0.0000, 0.7071))])
    dsub_plug = KeyPointCfg([Offset(pos=(0.0156, 0.0327, 0.0060), quat=(0.70711, 0.0000, 0.0000, 0.7071))])
    rj_45_plug = KeyPointCfg([Offset(pos=(-0.1500, 0.0821, 0.0060), quat=(0.70711, 0.0000, 0.0000, 0.7071))])
    waterproof_plug = KeyPointCfg([Offset(pos=(0.1345, -0.1051, -0.015), quat=(0.70711, 0.0000, 0.0000, 0.7071))])
    usba_plug = KeyPointCfg([Offset(pos=(-0.1834, -0.0906, 0.0060), quat=(0.7071, 0.0000, 0.7071, 0.0000))])

    nut_m4 = KeyPointCfg([Offset(pos=(0.1169, 0.1427, -0.0049), quat=(0.2588, 0.0000, 0.0000, 0.9659))])
    nut_m8 = KeyPointCfg([Offset(pos=(0.0646, -0.0052, -0.0091), quat=(0.25882, 0.0000, 0.0000, -0.96593))])
    nut_m12 = KeyPointCfg([Offset(pos=(-0.0112, -0.1362, -0.0130), quat=(0.25882, 0.0000, 0.0000, -0.96593))])
    nut_m16 = KeyPointCfg([Offset(pos=(-0.1078, -0.0177, 0.0118), quat=(0.25882, 0.0000, 0.0000, -0.96593))])

    rectangular_peg_4mm = KeyPointCfg([Offset(pos=(-0.1482, -0.1419, 0.0060), quat=(0.7071, 0.0000, 0.7071, 0.0000))])
    rectangular_peg_8mm = KeyPointCfg([Offset(pos=(-0.0899, 0.0836, 0.0060), quat=(1.0000, 0.0000, 0.0000, 0.0000))])
    rectangular_peg_12mm = KeyPointCfg([Offset(pos=(-0.0248, 0.1350, 0.0060), quat=(1.0000, 0.0000, 0.0000, 0.0000))])
    rectangular_peg_16mm = KeyPointCfg([Offset(pos=(0.1432, 0.0886, 0.005), quat=(0.7071, 0.0000, 0.0000, 0.7071))])

    rod_4mm = KeyPointCfg([Offset(pos=(0.1399, 0.0297, 0.010), quat=(1.0000, 0.0000, 0.0000, 0.0000))])
    rod_8mm = KeyPointCfg([Offset(pos=(-0.1408, 0.1306, 0.005), quat=(1.0000, 0.0000, 0.0000, 0.0000))])
    rod_12mm = KeyPointCfg([Offset(pos=(-0.0592, 0.0050, 0.005), quat=(-0.70711, 0.0000, 0.0000, 0.7071))])
    rod_16mm = KeyPointCfg([Offset(pos=(0.0704, -0.0740, 0.005), quat=(-0.70711, 0.0000, 0.0000, 0.7071))])

    large_gear = KeyPointCfg([Offset(pos=(0.0390, 0.0860, 0.0050), quat=(-0.70711, 0.0000, 0.0000, 0.7071))])
    medium_gear = KeyPointCfg([Offset(pos=(0.0061, -0.0475, 0.0050), quat=(0.7071, 0.0000, 0.0000, -0.70711))])
    small_gear = KeyPointCfg([Offset(pos=(0.0595, -0.0875, 0.0050), quat=(-0.70711, 0.0000, 0.0000, 0.7071))])


@configclass
class NistBoardKeyPointsCfg(ObjectKeyPointsCfg):
    bnc_plug = KeyPointCfg([Offset(pos=(0.2797, -0.1915, -0.0000), quat=(0.0000, 0.7071, 0.7071, 0.0000))])
    bnc_socket = KeyPointCfg([Offset(pos=(0.2797, -0.1915, 0.0000), quat=(0.0000, 0.7071, 0.7071, 0.0000))])
    dsub_plug = KeyPointCfg([Offset(pos=(0.2129, -0.2659, -0.0190), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    dsub_socket = KeyPointCfg([Offset(pos=(0.2129, -0.2659, -0.0190), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    rj_45_plug = KeyPointCfg([Offset(pos=(0.3473, -0.3415, 0.0000), quat=(0.0000, 0.0000, 1.0, 0.0000))])
    rj_45_socket = KeyPointCfg([Offset(pos=(0.3473, -0.3415, 0.0000), quat=(0.0000, 0.0000, 1.0, 0.0000))])
    waterproof_plug = KeyPointCfg([Offset(pos=(0.1981, -0.1166, -0.0002), quat=(0.0000, 0.0000, 1.0, 0.0000))])
    waterproof_socket = KeyPointCfg([Offset(pos=(0.1981, -0.1166, -0.0002), quat=(0.0000, 0.0000, 1.0, 0.0000))])
    usba_plug = KeyPointCfg([Offset(pos=(0.2721, -0.0415, -0.0001), quat=(0.0000, 0.7071, 0.7071, 0.0000))])
    usba_socket = KeyPointCfg([Offset(pos=(0.2721, -0.0415, -0.0001), quat=(0.0000, 0.7071, 0.7071, 0.0000))])

    nut_m4 = KeyPointCfg([Offset(pos=(0.1223, -0.1914, 0.0040), quat=(0.0000, 1.0, 0.0000, 0.0000))])
    bolt_m4 = KeyPointCfg([Offset(pos=(0.1223, -0.1914, 0.0131), quat=(0.0000, 1.0, 0.0000, 0.0000))])
    nut_m8 = KeyPointCfg([Offset(pos=(0.0473, -0.0407, 0.0085), quat=(0.0000, 1.0, 0.0000, 0.0000))])
    bolt_m8 = KeyPointCfg([Offset(pos=(0.0473, -0.0407, 0.0172), quat=(0.0000, 1.0, 0.0000, 0.0000))])
    nut_m12 = KeyPointCfg([Offset(pos=(0.3473, -0.2665, 0.0123), quat=(0.0000, 1.0, 0.0000, 0.0000))])
    bolt_m12 = KeyPointCfg([Offset(pos=(0.3473, -0.2665, 0.0212), quat=(0.0000, 1.0, 0.0000, 0.0000))])
    nut_m16 = KeyPointCfg([Offset(pos=(0.04715, -0.3416, 0.0094), quat=(0.0000, 1.0, 0.0000, 0.0000))])
    bolt_m16 = KeyPointCfg([Offset(pos=(0.04715, -0.3416, 0.0194), quat=(0.0000, 1.0, 0.0000, 0.0000))])

    rectangular_peg_4mm = KeyPointCfg([Offset(pos=(0.1971, -0.1915, -0.0003), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    rectangular_hole_4mm = KeyPointCfg([Offset(pos=(0.1971, -0.1915, -0.0003), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    rectangular_peg_8mm = KeyPointCfg([Offset(pos=(0.2717, -0.2659, -0.0003), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    rectangular_hole_8mm = KeyPointCfg([Offset(pos=(0.2717, -0.2659, -0.0003), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    rectangular_peg_12mm = KeyPointCfg([Offset(pos=(0.1971, -0.0413, -0.0003), quat=(0.0000, 0.0000, 1.0, 0.0000))])
    rectangular_hole_12mm = KeyPointCfg([Offset(pos=(0.1971, -0.0413, -0.0003), quat=(0.0000, 0.0000, 1.0, 0.0000))])
    rectangular_peg_16mm = KeyPointCfg([Offset(pos=(0.3472, -0.0413, -0.0003), quat=(0.0000, 0.7071, 0.7071, 0.0000))])
    rectangular_hole_16mm = KeyPointCfg([Offset(pos=(0.3472, -0.0413, -0.0003), quat=(0.0000, 0.7071, 0.7071, 0.0000))])

    rod_4mm = KeyPointCfg([Offset(pos=(0.3473, -0.1918, -0.0001), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    hole_4mm = KeyPointCfg([Offset(pos=(0.3473, -0.1918, -0.0001), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    rod_8mm = KeyPointCfg([Offset(pos=(0.3473, -0.1164, -0.0001), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    hole_8mm = KeyPointCfg([Offset(pos=(0.3473, -0.1164, -0.0001), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    rod_12mm = KeyPointCfg([Offset(pos=(0.1226, -0.0422, -0.0001), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    hole_12mm = KeyPointCfg([Offset(pos=(0.1226, -0.0422, -0.0001), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    rod_16mm = KeyPointCfg([Offset(pos=(0.1221, -0.2665, -0.0001), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    hole_16mm = KeyPointCfg([Offset(pos=(0.1221, -0.2665, -0.0001), quat=(0.0000, 0.7071, -0.7071, 0.0000))])

    large_gear = KeyPointCfg([Offset(pos=(0.0474, -0.1713, -0.0002), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    medium_gear = KeyPointCfg([Offset(pos=(0.0459, -0.1714, -0.0002), quat=(0.0000, 0.73566, -0.67736, 0.0000))])
    small_gear = KeyPointCfg([Offset(pos=(0.0474, -0.1713, -0.0002), quat=(0.0000, 0.7071, -0.7071, 0.0000))])
    gear_base = KeyPointCfg([Offset(pos=(0.0474, -0.1713, -0.0002), quat=(0.0000, 0.7071, -0.7071, 0.0000))])

    nist_board_center = KeyPointCfg([Offset(pos=(0.197176, -0.19145, 0.0000))])


@configclass
class BoltM16KeyPointsCfg(ObjectKeyPointsCfg):
    tip = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0350))])
    one_cm_above_tip = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0450))])
    full_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0100))])
    first_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.034))])
    second_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0320))])
    third_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0300))])
    eighth_thread_nist_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0200))])


def xy_continuous_symmetry(origin, offset) -> list[SymmetryOffsets]:
    axis = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    angles = [[0.0, 3.1415], [0.0, 3.1415], []]
    s = SymmetryOffsets(origin=origin, angles=angles, axis=axis, offset=offset)
    return [s]


def xy_2fold_symmetry(origin, offset) -> list[SymmetryOffsets]:
    axis = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    angles = [[0.0, 3.1415], [0.0, 3.1415], [0.0, 3.1415]]
    s = SymmetryOffsets(origin=origin, angles=angles, axis=axis, offset=offset)
    return [s]


def xy_4fold_symmetry(origin, offset) -> list[SymmetryOffsets]:
    axis = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    angles = [[0.0, 3.1415], [0.0, 3.1415], [0.0, 1.57, 3.1415, 4.7124]]
    s = SymmetryOffsets(origin=origin, angles=angles, axis=axis, offset=offset)
    return [s]


def z_axis_continuous_symmetry(origin, offset) -> list[SymmetryOffsets]:
    axis = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    angles = [[0.0], [0.0], []]
    s = SymmetryOffsets(origin=origin, angles=angles, axis=axis, offset=offset)
    return [s]


@configclass
class NutM16KeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0165))
    nut_opening = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0100))))
    grasp_point = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0100), quat=(0.7071, 0.0000, 0.0000, -0.7071))))
    grasp_diameter: float = 0.0240


@configclass
class BoltM12KeyPointsCfg(ObjectKeyPointsCfg):
    tip = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0350))])
    head = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])
    second_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0285))])
    seventh_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0215))])


@configclass
class NutM12KeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0180))
    nut_opening = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0130))))
    grasp_point = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0130), quat=(0.7071, 0.0000, 0.0000, -0.7071))))
    grasp_diameter: float = 0.019


@configclass
class BoltM8KeyPointsCfg(ObjectKeyPointsCfg):
    tip = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.02600))])
    head = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])
    full_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0084))])
    second_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0242))])
    seventh_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0182))])


@configclass
class NutM8KeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0126))
    nut_opening = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0093))))
    grasp_point = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0093), quat=(0.7071, 0.0000, 0.0000, -0.7071))))
    grasp_diameter: float = 0.013


@configclass
class BoltM4KeyPointsCfg(ObjectKeyPointsCfg):
    tip = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0200))])
    head = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])
    full_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0044))])
    second_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0189))])
    tenth_thread = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0134))])


@configclass
class NutM4KeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0064))
    nut_opening = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0046))))
    grasp_point = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0046), quat=(0.7071, 0.0000, 0.0000, -0.7071))))
    grasp_diameter: float = 0.0070


@configclass
class GearBaseKeyPointsCfg(ObjectKeyPointsCfg):
    small_gear_shaft_tip = KeyPointCfg([Offset(pos=(0.0508, 0.0000, 0.0250))])
    small_gear_shaft_bottom = KeyPointCfg([Offset(pos=(0.05075, 0.0000, 0.0050))])
    medium_gear_shaft_tip = KeyPointCfg([Offset(pos=(0.02025, 0.0000, 0.0250))])
    medium_gear_shaft_bottom = KeyPointCfg([Offset(pos=(0.02025, 0.0000, 0.0050))])
    large_gear_shaft_tip = KeyPointCfg([Offset(pos=(-0.0303, 0.0000, 0.0250))])
    large_gear_shaft_bottom = KeyPointCfg([Offset(pos=(-0.0303, 0.0000, 0.0050))])


@configclass
class SmallGearKeyPointsCfg(ObjectKeyPointsCfg):
    center = Offset(pos=(0.05075, 0.0000, 0.0220))
    center_axis_bottom = KeyPointCfg(z_axis_continuous_symmetry(center, offset=Offset(pos=(0.05075, 0.0000, 0.0050))))
    center_axis_top = KeyPointCfg([Offset(pos=(0.05075, 0.0000, 0.0300))])
    grasp_point = KeyPointCfg([Offset(pos=(0.05075, 0.0000, 0.0220))])
    grasp_diameter: float = 0.0175


@configclass
class MediumGearKeyPointsCfg(ObjectKeyPointsCfg):
    center = Offset(pos=(0.02025, 0.0000, 0.0220))
    center_axis_top = KeyPointCfg([Offset(pos=(0.02025, 0.0000, 0.0300))])
    grasp_point = KeyPointCfg([Offset(pos=(0.02025, 0.0000, 0.0220))])
    center_axis_bottom = KeyPointCfg(z_axis_continuous_symmetry(center, offset=Offset(pos=(0.02025, 0.0000, 0.0050))))
    grasp_diameter: float = 0.0300


@configclass
class LargeGearKeyPointsCfg(ObjectKeyPointsCfg):
    center = Offset(pos=(-0.0303, 0.0000, 0.0300))
    center_axis_top = KeyPointCfg([Offset(pos=(-0.0303, 0.0000, 0.0300))])
    grasp_point = KeyPointCfg([Offset(pos=(-0.0303, 0.0000, 0.0220))])
    center_axis_bottom = KeyPointCfg(z_axis_continuous_symmetry(center, offset=Offset(pos=(-0.0303, 0.0000, 0.0050))))
    grasp_diameter: float = 0.0300


@configclass
class Hole16MMKeyPointsCfg(ObjectKeyPointsCfg):
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0090))])
    hole_bottom = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])


@configclass
class Rod16MMKeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0250))
    rod_tip = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0000))))
    grasp_point = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0350))))
    grasp_diameter: float = 0.0160


@configclass
class Hole12MMKeyPointsCfg(ObjectKeyPointsCfg):
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0090))])
    hole_bottom = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])


@configclass
class Rod12MMKeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0250))
    rod_tip = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0000))))
    grasp_point = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0350))))
    grasp_diameter: float = 0.0120


@configclass
class Hole8MMKeyPointsCfg(ObjectKeyPointsCfg):
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0090))])
    hole_bottom = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])


@configclass
class Rod8MMKeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0250))
    rod_tip = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0000))))
    grasp_point = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0350))))
    grasp_diameter: float = 0.0080


@configclass
class Hole4MMKeyPointsCfg(ObjectKeyPointsCfg):
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0090))])
    hole_bottom = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])


@configclass
class Rod4MMKeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0250))
    rod_tip = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0000))))
    grasp_point = KeyPointCfg(xy_continuous_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0350))))
    grasp_diameter: float = 0.0040


@configclass
class RectangularPeg16MMKeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0250))
    peg_tip = KeyPointCfg(xy_2fold_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0000))))
    grasp_point = KeyPointCfg(xy_2fold_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0350))))
    grasp_diameter: float = 0.0100  # y axis is 0.01 while x axis is 0.016, robot gripper is y axis


@configclass
class RectangularHole16MMKeyPointsCfg(ObjectKeyPointsCfg):
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0090))])
    hole_bottom = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])


@configclass
class RectangularPeg12MMKeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0250))
    peg_tip = KeyPointCfg(xy_2fold_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0000))))
    grasp_point = KeyPointCfg(xy_2fold_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0350))))
    grasp_diameter: float = 0.0080  # y axis is 0.008 while x axis is 0.012, robot gripper is y axis


@configclass
class RectangularHole12MMKeyPointsCfg(ObjectKeyPointsCfg):
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0090))])
    hole_bottom = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])


@configclass
class RectangularPeg8MMKeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0250))
    peg_tip = KeyPointCfg(xy_2fold_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0000))))
    grasp_point = KeyPointCfg(xy_2fold_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0350))))
    grasp_diameter: float = 0.0080


@configclass
class RectangularHole8MMKeyPointsCfg(ObjectKeyPointsCfg):
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0090))])
    hole_bottom = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])


@configclass
class RectangularPeg4MMKeyPointsCfg(ObjectKeyPointsCfg):
    geometry_origin = Offset(pos=(0.0000, 0.0000, 0.0250))
    peg_tip = KeyPointCfg(xy_4fold_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0000))))
    grasp_point = KeyPointCfg(xy_4fold_symmetry(geometry_origin, offset=Offset(pos=(0.0000, 0.0000, 0.0350))))
    grasp_diameter: float = 0.0040


@configclass
class RectangularHole4MMKeyPointsCfg(ObjectKeyPointsCfg):
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0090))])
    hole_bottom = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.000))])


@configclass
class USBAPlugKeyPointsCfg(ObjectKeyPointsCfg):
    insertion_tip = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0335))])
    tail = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0930))])
    grasp_point = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0563))])
    grasp_diameter: float = 0.0152  # 0.0076 for grasp at thin part


@configclass
class USBASocketKeyPointsCfg(ObjectKeyPointsCfg):
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0416))])
    hole_bottom = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0335))])
    housing_bottom = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])


@configclass
class WaterproofPlugKeyPointsCfg(ObjectKeyPointsCfg):
    insertion_tip = KeyPointCfg([Offset(pos=(0.0, 0.0, 0.021))])
    tail = KeyPointCfg([Offset(pos=(0.0, 0.0, 0.0589))])
    grasp_point = KeyPointCfg([Offset(pos=(0.0, 0.0, 0.0440))])
    grasp_diameter: float = 0.03152


@configclass
class WaterproofSocketKeyPointsCfg(ObjectKeyPointsCfg):
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0340))])
    plug_assembled = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0210))])


@configclass
class BNCPlugKeyPointsCfg(ObjectKeyPointsCfg):
    insertion_tip = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0212))])
    grasp_point = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0197))])
    plug_entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0109), quat=(0.7071, 0.0000, 0.0000, 0.7071))])
    grasp_diameter: float = 0.0143


@configclass
class BNCSocketKeyPointsCfg(ObjectKeyPointsCfg):
    plug_assembled = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0212))])


@configclass
class DSUBPlugKeyPointsCfg(ObjectKeyPointsCfg):
    insertion_tip = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])
    grasp_point = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0125))])
    grasp_diameter: float = 0.013


@configclass
class DSUBSocketKeyPointsCfg(ObjectKeyPointsCfg):
    plug_assembled = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0000))])
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0061))])


@configclass
class RJ45PlugKeyPointsCfg(ObjectKeyPointsCfg):
    insertion_tip = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0150))])
    tail = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0771))])
    grasp_point = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0388))])
    grasp_diameter: float = 0.0110


@configclass
class RJ45SocketKeyPointsCfg(ObjectKeyPointsCfg):
    entry = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0280))])
    plug_assembled = KeyPointCfg([Offset(pos=(0.0000, 0.0000, 0.0150))])


@configclass
class PandaHandKeyPointsCfg(ObjectKeyPointsCfg):
    gripper_center_grasp_point = KeyPointCfg([Offset(pos=(0.0, 0.0, 0.107), quat=(0.0, 0.0, 1.0, 0.0))])
    gripper_tip_grasp_point = KeyPointCfg([Offset(pos=(0.0, 0.0, 0.112), quat=(0.0, 0.0, 1.0, 0.0))])


@configclass
class RobotRootKeyPointsCfg(ObjectKeyPointsCfg):
    base = KeyPointCfg([Offset(pos=(0.0, 0.0, 0.0), quat=(0.0, 0.0, 1.0, 0.0))])


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

# ROBOT KEY_POINTS
KEY_POINTS_PANDA_HAND = PandaHandKeyPointsCfg()
KEY_POINTS_ROBOT = RobotRootKeyPointsCfg()
