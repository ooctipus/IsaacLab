from __future__ import annotations
from isaaclab.utils import configclass

from . import assembly_object_key_points as kps
from . import task_definition_cfg

KpCfg = task_definition_cfg.KeyPointCfg


@configclass
class NutM16(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.NUT_M16_KEY_POINTS.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_tip_grasp_point,
        asset_grasp=kps.NUT_M16_KEY_POINTS.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rp', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class NutM12(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.NUT_M12_KEY_POINTS.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_tip_grasp_point,
        asset_grasp=kps.NUT_M12_KEY_POINTS.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rp', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class NutM8(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.NUT_M8_KEY_POINTS.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_tip_grasp_point,
        asset_grasp=kps.NUT_M8_KEY_POINTS.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rp', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class NutM4(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.NUT_M4_KEY_POINTS.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_tip_grasp_point,
        asset_grasp=kps.NUT_M4_KEY_POINTS.grasp_point,
    )


@configclass
class Rod16MM(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_ROD_16MM.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_ROD_16MM.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rp', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class Rod12MM(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_ROD_12MM.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_ROD_12MM.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rp', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class Rod8MM(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_ROD_8MM.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_ROD_8MM.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rp', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class Rod4MM(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_ROD_4MM.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_ROD_4MM.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rp', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class RectangularPeg16MM(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_RECTANGULAR_PEG_16MM.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_RECTANGULAR_PEG_16MM.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rpy', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class RectangularPeg12MM(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_RECTANGULAR_PEG_12MM.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_RECTANGULAR_PEG_12MM.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rpy', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class RectangularPeg8MM(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_RECTANGULAR_PEG_8MM.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_RECTANGULAR_PEG_8MM.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rpy', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class RectangularPeg4MM(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_RECTANGULAR_PEG_4MM.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_RECTANGULAR_PEG_4MM.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rpy', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class LargeGear(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_LARGE_GEAR.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_LARGE_GEAR.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rp', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class MediumGear(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_MEDIUM_GEAR.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_MEDIUM_GEAR.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rp', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class SmallGear(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_SMALL_GEAR.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_SMALL_GEAR.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rp', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class USBAPlug(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_USB_A_PLUG.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_USB_A_PLUG.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rpy', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class WaterproofPlug(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_WATERPROOF_PLUG.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_WATERPROOF_PLUG.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rpy', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class DSUBPlug(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_D_SUB_PLUG.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_D_SUB_PLUG.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rpy', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class BNCPlug(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_BNC_PLUG.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_BNC_PLUG.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rpy', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )


@configclass
class RJ45Plug(task_definition_cfg.ManipulationKeyPointCfg):
    held_asset_diameter = kps.KEY_POINTS_RJ45_PLUG.grasp_diameter
    key_points = task_definition_cfg.ManipulationKeyPointCfg.ManipulationKeyPoints(
        robot_root=kps.KEY_POINTS_ROBOT.base,
        robot_object_held=kps.KEY_POINTS_PANDA_HAND.gripper_center_grasp_point,
        asset_grasp=kps.KEY_POINTS_RJ45_PLUG.grasp_point,
    )
    success_condition = task_definition_cfg.SuccessCondition(
        rot_components='rpy', pos_threshold=(0.02, 0.02, 0.02), rot_threshold=(0.25, 0.25, 0.25)
    )
