from __future__ import annotations
from isaaclab.utils import configclass

from . import assembly_object_key_points as kps
from . import task_definition_cfg


@configclass
class NutThreadM16(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.NUT_M16_KEY_POINTS.nut_opening,
        asset_align_against=kps.BOLT_M16_KEY_POINTS.tip,
    )
    

@configclass
class NutThreadM16SecondThread(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.NUT_M16_KEY_POINTS.nut_opening,
        asset_align_against=kps.BOLT_M16_KEY_POINTS.tip,
    )


@configclass
class NutThreadM12(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.NUT_M12_KEY_POINTS.nut_opening,
        asset_align_against=kps.BOLT_M12_KEY_POINTS.tip,
    )


@configclass
class NutThreadM8(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.NUT_M8_KEY_POINTS.nut_opening,
        asset_align_against=kps.BOLT_M8_KEY_POINTS.tip,
    )


@configclass
class NutThreadM4(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.NUT_M4_KEY_POINTS.nut_opening,
        asset_align_against=kps.BOLT_M4_KEY_POINTS.tip,
    )


@configclass
class Rod16MMInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_ROD_16MM.rod_tip,
        asset_align_against=kps.KEY_POINTS_HOLE_16MM.entry,
    )


@configclass
class Rod12MMInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_ROD_12MM.rod_tip,
        asset_align_against=kps.KEY_POINTS_HOLE_12MM.entry,
    )


@configclass
class Rod8MMInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_ROD_8MM.rod_tip,
        asset_align_against=kps.KEY_POINTS_HOLE_8MM.entry,
    )


@configclass
class Rod4MMInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_ROD_4MM.rod_tip,
        asset_align_against=kps.KEY_POINTS_HOLE_4MM.entry,
    )


@configclass
class RectangularPeg16MMInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_RECTANGULAR_PEG_16MM.peg_tip,
        asset_align_against=kps.KEY_POINTS_RECTANGULAR_HOLE_16MM.entry,
    )


@configclass
class RectangularPeg12MMInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_RECTANGULAR_PEG_12MM.peg_tip,
        asset_align_against=kps.KEY_POINTS_RECTANGULAR_HOLE_12MM.entry,
    )


@configclass
class RectangularPeg8MMInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_RECTANGULAR_PEG_8MM.peg_tip,
        asset_align_against=kps.KEY_POINTS_RECTANGULAR_HOLE_8MM.entry,
    )


@configclass
class RectangularPeg4MMInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_RECTANGULAR_PEG_4MM.peg_tip,
        asset_align_against=kps.KEY_POINTS_RECTANGULAR_HOLE_4MM.entry,
    )


@configclass
class GearMeshLarge(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_LARGE_GEAR.center_axis_bottom,
        asset_align_against=kps.KEY_POINTS_GEAR_BASE.large_gear_shaft_tip,
    )


@configclass
class GearMeshMedium(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_MEDIUM_GEAR.center_axis_bottom,
        asset_align_against=kps.KEY_POINTS_GEAR_BASE.medium_gear_shaft_tip,
    )


@configclass
class GearMeshSmall(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_SMALL_GEAR.center_axis_bottom,
        asset_align_against=kps.KEY_POINTS_GEAR_BASE.small_gear_shaft_tip,
    )


@configclass
class USBAInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_USB_A_PLUG.insertion_tip,
        asset_align_against=kps.KEY_POINTS_USB_A_SOCKET.entry,
    )


@configclass
class WaterproofInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_WATERPROOF_PLUG.insertion_tip,
        asset_align_against=kps.KEY_POINTS_WATERPROOF_SOCKET.entry,
    )


@configclass
class DSUBInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_D_SUB_PLUG.insertion_tip,
        asset_align_against=kps.KEY_POINTS_D_SUB_SOCKET.entry,
    )


@configclass
class BNCInsert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_BNC_PLUG.plug_entry,
        asset_align_against=kps.KEY_POINTS_BNC_SOCKET.plug_assembled,
    )


@configclass
class RJ45Insert(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_RJ45_PLUG.insertion_tip,
        asset_align_against=kps.KEY_POINTS_RJ45_SOCKET.entry,
    )


# Alignment tasks
@configclass
class NutThreadM16Align(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.NUT_M16_KEY_POINTS.nut_opening,
        asset_align_against=kps.BOLT_M16_KEY_POINTS.tip,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rp")


@configclass
class NutThreadM12Align(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.NUT_M12_KEY_POINTS.nut_opening,
        asset_align_against=kps.BOLT_M12_KEY_POINTS.tip,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rp")


@configclass
class NutThreadM8Align(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.NUT_M8_KEY_POINTS.nut_opening,
        asset_align_against=kps.BOLT_M8_KEY_POINTS.tip,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rp")


@configclass
class NutThreadM4Align(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.NUT_M4_KEY_POINTS.nut_opening,
        asset_align_against=kps.BOLT_M4_KEY_POINTS.tip,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rp")


@configclass
class Rod16MMAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_ROD_16MM.rod_tip,
        asset_align_against=kps.KEY_POINTS_HOLE_16MM.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rp")


@configclass
class Rod12MMAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_ROD_12MM.rod_tip,
        asset_align_against=kps.KEY_POINTS_HOLE_12MM.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rp")


@configclass
class Rod8MMAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_ROD_8MM.rod_tip,
        asset_align_against=kps.KEY_POINTS_HOLE_8MM.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rp")


@configclass
class Rod4MMAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_ROD_4MM.rod_tip,
        asset_align_against=kps.KEY_POINTS_HOLE_4MM.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rp")


@configclass
class RectangularPeg16MMAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_RECTANGULAR_PEG_16MM.peg_tip,
        asset_align_against=kps.KEY_POINTS_RECTANGULAR_HOLE_16MM.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rpy")


@configclass
class RectangularPeg12MMAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_RECTANGULAR_PEG_12MM.peg_tip,
        asset_align_against=kps.KEY_POINTS_RECTANGULAR_HOLE_12MM.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rpy")


@configclass
class RectangularPeg8MMAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_RECTANGULAR_PEG_8MM.peg_tip,
        asset_align_against=kps.KEY_POINTS_RECTANGULAR_HOLE_8MM.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rpy")


@configclass
class RectangularPeg4MMAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_RECTANGULAR_PEG_4MM.peg_tip,
        asset_align_against=kps.KEY_POINTS_RECTANGULAR_HOLE_4MM.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rpy")


@configclass
class GearLargeAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_LARGE_GEAR.center_axis_bottom,
        asset_align_against=kps.KEY_POINTS_GEAR_BASE.large_gear_shaft_tip,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rp")


@configclass
class GearMediumAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_MEDIUM_GEAR.center_axis_bottom,
        asset_align_against=kps.KEY_POINTS_GEAR_BASE.medium_gear_shaft_tip,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rp")


@configclass
class GearSmallAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_SMALL_GEAR.center_axis_bottom,
        asset_align_against=kps.KEY_POINTS_GEAR_BASE.small_gear_shaft_tip,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rp")


@configclass
class USBAAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_USB_A_PLUG.insertion_tip,
        asset_align_against=kps.KEY_POINTS_USB_A_SOCKET.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rpy")


@configclass
class WaterproofAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_WATERPROOF_PLUG.insertion_tip,
        asset_align_against=kps.KEY_POINTS_WATERPROOF_SOCKET.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rpy")


@configclass
class DSUBAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_D_SUB_PLUG.insertion_tip,
        asset_align_against=kps.KEY_POINTS_D_SUB_SOCKET.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rpy")


@configclass
class BNCAlign(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_BNC_PLUG.plug_entry,
        asset_align_against=kps.KEY_POINTS_BNC_SOCKET.plug_assembled,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rpy")


@configclass
class RJ45Align(task_definition_cfg.TaskKeyPointCfg):
    key_points = task_definition_cfg.TaskKeyPointCfg.TaskKeyPoints(
        asset_align=kps.KEY_POINTS_RJ45_PLUG.insertion_tip,
        asset_align_against=kps.KEY_POINTS_RJ45_SOCKET.entry,
    )
    success_condition = task_definition_cfg.SuccessCondition(rot_components="rpy")
