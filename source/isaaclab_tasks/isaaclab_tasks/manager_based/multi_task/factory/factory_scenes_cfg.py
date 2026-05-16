# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene definitions for all 20 Factory task variants.

Each scene class inherits from :class:`FactorySceneBase` and specifies the
``fixed_asset`` / ``held_asset`` pair (plus extra scene entities for gear tasks).
:class:`FactorySceneCfg` is the :class:`PresetCfg` that selects among them.
"""

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import factory_assets_cfg as assets
from .mdp_presets import (
    GripperLeftContactSensorCfg,
    GripperRightContactSensorCfg,
    RobotArticulationCfg,
)


@configclass
class FactorySceneBase(InteractiveSceneCfg):
    """Shared scene assets for all Factory tasks.

    The ``robot`` and gripper-finger contact sensors are resolved from the
    active robot preset (e.g. ``presets=franka``); see
    :mod:`.mdp_presets.robots` for how to add a robot.
    """

    ground = assets.GROUND_CFG
    table = assets.TABLE_CFG
    nistboard = assets.NISTBOARD_CFG
    robot: ArticulationCfg = RobotArticulationCfg()  # type: ignore
    panda_leftfinger_object_s: ContactSensorCfg | None = GripperLeftContactSensorCfg()  # type: ignore
    panda_rightfinger_object_s: ContactSensorCfg | None = GripperRightContactSensorCfg()  # type: ignore
    dome_light = assets.DOMELIGHT_CFG


# ---------------------------------------------------------------------------
# Nut threading (4 sizes)
# ---------------------------------------------------------------------------


@configclass
class NutThreadM4SceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.BOLT_M4_CFG
    held_asset: RigidObjectCfg = assets.NUT_M4_CFG


@configclass
class NutThreadM8SceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.BOLT_M8_CFG
    held_asset: RigidObjectCfg = assets.NUT_M8_CFG


@configclass
class NutThreadM12SceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.BOLT_M12_CFG
    held_asset: RigidObjectCfg = assets.NUT_M12_CFG


@configclass
class NutThreadM16SceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.BOLT_M16_CFG
    held_asset: RigidObjectCfg = assets.NUT_M16_CFG


# ---------------------------------------------------------------------------
# Gear mesh (3 sizes — all share the same base with 3 gear shafts)
# ---------------------------------------------------------------------------


@configclass
class GearMeshSmallSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.GEAR_BASE_CFG
    held_asset: RigidObjectCfg = assets.SMALL_GEAR_CFG
    medium_gear: RigidObjectCfg = assets.MEDIUM_GEAR_CFG
    large_gear: RigidObjectCfg = assets.LARGE_GEAR_CFG


@configclass
class GearMeshMediumSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.GEAR_BASE_CFG
    held_asset: RigidObjectCfg = assets.MEDIUM_GEAR_CFG
    small_gear: RigidObjectCfg = assets.SMALL_GEAR_CFG
    large_gear: RigidObjectCfg = assets.LARGE_GEAR_CFG


@configclass
class GearMeshLargeSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.GEAR_BASE_CFG
    held_asset: RigidObjectCfg = assets.LARGE_GEAR_CFG
    small_gear: RigidObjectCfg = assets.SMALL_GEAR_CFG
    medium_gear: RigidObjectCfg = assets.MEDIUM_GEAR_CFG


# ---------------------------------------------------------------------------
# Rod insert — round (4 sizes)
# ---------------------------------------------------------------------------


@configclass
class RodInsert4MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.HOLE_4MM_CFG
    held_asset: RigidObjectCfg = assets.ROD_4MM_CFG


@configclass
class RodInsert8MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.HOLE_8MM_CFG
    held_asset: RigidObjectCfg = assets.ROD_8MM_CFG


@configclass
class RodInsert12MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.HOLE_12MM_CFG
    held_asset: RigidObjectCfg = assets.ROD_12MM_CFG


@configclass
class RodInsert16MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.HOLE_16MM_CFG
    held_asset: RigidObjectCfg = assets.ROD_16MM_CFG


# ---------------------------------------------------------------------------
# Peg insert — rectangular (4 sizes)
# ---------------------------------------------------------------------------


@configclass
class PegInsert4MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.RECTANGULAR_HOLE_4MM_CFG
    held_asset: RigidObjectCfg = assets.RECTANGULAR_PEG_4MM_CFG


@configclass
class PegInsert8MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.RECTANGULAR_HOLE_8MM_CFG
    held_asset: RigidObjectCfg = assets.RECTANGULAR_PEG_8MM_CFG


@configclass
class PegInsert12MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.RECTANGULAR_HOLE_12MM_CFG
    held_asset: RigidObjectCfg = assets.RECTANGULAR_PEG_12MM_CFG


@configclass
class PegInsert16MMSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.RECTANGULAR_HOLE_16MM_CFG
    held_asset: RigidObjectCfg = assets.RECTANGULAR_PEG_16MM_CFG


# ---------------------------------------------------------------------------
# Connector insert (5 types)
# ---------------------------------------------------------------------------


@configclass
class ConnectorUSBASceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.USBA_SOCKET_CFG
    held_asset: RigidObjectCfg = assets.USBA_PLUG_CFG


@configclass
class ConnectorWaterproofSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.WATERPROOF_SOCKET_CFG
    held_asset: RigidObjectCfg = assets.WATERPROOF_PLUG_CFG


@configclass
class ConnectorBNCSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.BNC_SOCKET_CFG
    held_asset: RigidObjectCfg = assets.BNC_PLUG_CFG


@configclass
class ConnectorDSUBSceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.DSUB_SOCKET_CFG
    held_asset: RigidObjectCfg = assets.DSUB_PLUG_CFG


@configclass
class ConnectorRJ45SceneCfg(FactorySceneBase):
    fixed_asset: RigidObjectCfg = assets.RJ45_SOCKET_CFG
    held_asset: RigidObjectCfg = assets.RJ45_PLUG_CFG


@configclass
class FactorySceneCfg(PresetCfg):
    """Task scene preset — resolves to the complete scene for the active task."""

    # Nut threading
    nut_thread_m4: NutThreadM4SceneCfg = NutThreadM4SceneCfg(num_envs=2, env_spacing=2.0)
    nut_thread_m8: NutThreadM8SceneCfg = NutThreadM8SceneCfg(num_envs=2, env_spacing=2.0)
    nut_thread_m12: NutThreadM12SceneCfg = NutThreadM12SceneCfg(num_envs=2, env_spacing=2.0)
    nut_thread_m16: NutThreadM16SceneCfg = NutThreadM16SceneCfg(num_envs=2, env_spacing=2.0)

    # Gear mesh
    gear_mesh_small: GearMeshSmallSceneCfg = GearMeshSmallSceneCfg(num_envs=2, env_spacing=2.0)
    gear_mesh_medium: GearMeshMediumSceneCfg = GearMeshMediumSceneCfg(num_envs=2, env_spacing=2.0)
    gear_mesh_large: GearMeshLargeSceneCfg = GearMeshLargeSceneCfg(num_envs=2, env_spacing=2.0)

    # Rod insert (round)
    rod_insert_4mm: RodInsert4MMSceneCfg = RodInsert4MMSceneCfg(num_envs=2, env_spacing=2.0)
    rod_insert_8mm: RodInsert8MMSceneCfg = RodInsert8MMSceneCfg(num_envs=2, env_spacing=2.0)
    rod_insert_12mm: RodInsert12MMSceneCfg = RodInsert12MMSceneCfg(num_envs=2, env_spacing=2.0)
    rod_insert_16mm: RodInsert16MMSceneCfg = RodInsert16MMSceneCfg(num_envs=2, env_spacing=2.0)

    # Peg insert (rectangular)
    peg_insert_4mm: PegInsert4MMSceneCfg = PegInsert4MMSceneCfg(num_envs=2, env_spacing=2.0)
    peg_insert_8mm: PegInsert8MMSceneCfg = PegInsert8MMSceneCfg(num_envs=2, env_spacing=2.0)
    peg_insert_12mm: PegInsert12MMSceneCfg = PegInsert12MMSceneCfg(num_envs=2, env_spacing=2.0)
    peg_insert_16mm: PegInsert16MMSceneCfg = PegInsert16MMSceneCfg(num_envs=2, env_spacing=2.0)

    # Connector insert
    usba: ConnectorUSBASceneCfg = ConnectorUSBASceneCfg(num_envs=2, env_spacing=2.0)
    waterproof: ConnectorWaterproofSceneCfg = ConnectorWaterproofSceneCfg(num_envs=2, env_spacing=2.0)
    bnc: ConnectorBNCSceneCfg = ConnectorBNCSceneCfg(num_envs=2, env_spacing=2.0)
    dsub: ConnectorDSUBSceneCfg = ConnectorDSUBSceneCfg(num_envs=2, env_spacing=2.0)
    rj45: ConnectorRJ45SceneCfg = ConnectorRJ45SceneCfg(num_envs=2, env_spacing=2.0)

    default: NutThreadM16SceneCfg = nut_thread_m16
