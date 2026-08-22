# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-safe full-board Factory scene."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.nist import factory_assets_cfg as assets
from isaaclab_tasks.contrib.nist.assembly_variants import ASSEMBLY_VARIANTS

from .board_layout import HELD_ASSET_NAMES, UNIQUE_FIXED_ASSET_NAMES, UNIQUE_FIXED_VARIANT_INDICES, clone_variant_rows


def _quat_mul(lhs: tuple[float, ...], rhs: tuple[float, ...]) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quat_apply(quat: tuple[float, ...], pos: tuple[float, ...]) -> tuple[float, float, float]:
    x, y, z, w = quat
    px, py, pz = pos
    tx, ty, tz = 2.0 * (y * pz - z * py), 2.0 * (z * px - x * pz), 2.0 * (x * py - y * px)
    return (
        px + w * tx + y * tz - z * ty,
        py + w * ty + z * tx - x * tz,
        pz + w * tz + x * ty - y * tx,
    )


_BOARD_INITIAL_STATE = assets.NISTBOARD_CFG.init_state
_fixed_asset_cfgs: list[RigidObjectCfg] = []
for _variant_index, _name in zip(UNIQUE_FIXED_VARIANT_INDICES, UNIQUE_FIXED_ASSET_NAMES, strict=True):
    _variant = ASSEMBLY_VARIANTS[_variant_index]
    _offset_pos = _quat_apply(_BOARD_INITIAL_STATE.rot, _variant.board_offset.pos)
    _fixed_asset_cfgs.append(
        _variant.fixed_asset.replace(
            prim_path=f"{{ENV_REGEX_NS}}/{_name}",
            spawn=_variant.fixed_asset.spawn.replace(
                fix_root_link=True,
                collision_props=assets.ASSEMBLY_SOCKET_COLLISION_PROPS_CFG.newton_mjwarp,
                physics_material=assets.ASSEMBLY_CONTACT_MATERIAL_CFG.newton_mjwarp,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(a + b for a, b in zip(_BOARD_INITIAL_STATE.pos, _offset_pos, strict=True)),
                rot=_quat_mul(_BOARD_INITIAL_STATE.rot, _variant.board_offset.quat),
            ),
        )
    )
_FIXED_ASSET_CFGS = tuple(_fixed_asset_cfgs)

_HELD_ASSET_SPAWNS = tuple(
    variant.held_asset.spawn.replace(
        collision_props=assets.ASSEMBLY_PLUG_COLLISION_PROPS_CFG.newton_mjwarp,
        physics_material=assets.ASSEMBLY_CONTACT_MATERIAL_CFG.newton_mjwarp,
    )
    for variant in ASSEMBLY_VARIANTS
)
_HELD_ASSET_CFGS = tuple(
    RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[spawn.copy() for spawn in _HELD_ASSET_SPAWNS],
            activate_contact_sensors=True,
            random_choice=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0 + 0.1 * slot)),
        mesh_variants_enabled=True,
        mesh_variant_inertia_diagonal_offset=1.0e-5,
    )
    for slot, name in enumerate(HELD_ASSET_NAMES)
)


@configclass
class FactoryBoardSceneCfg(InteractiveSceneCfg):
    """NIST board with every fixture and twenty homogeneous held bodies."""

    num_envs: int = 4096
    env_spacing: float = 2.0
    clone_cfg: cloner.CloneCfg = cloner.CloneCfg(valid_set=clone_variant_rows())

    ground = assets.GROUND_CFG
    table = assets.TABLE_CFG.replace(spawn=assets.TABLE_CFG.spawn.replace(fix_root_link=True))
    nistboard = assets.NISTBOARD_CFG.replace(
        spawn=assets.NISTBOARD_CFG.spawn.replace(
            fix_root_link=True,
            physics_material=assets.ASSEMBLY_CONTACT_MATERIAL_CFG.newton_mjwarp,
        )
    )
    robot = assets.FRANKA_PANDA_NEWTON_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=assets.FRANKA_PANDA_NEWTON_CFG.spawn.replace(
            physics_material=assets.ROBOT_CONTACT_MATERIAL_CFG.newton_mjwarp,
        ),
    )

    fixed_bolt_m4 = _FIXED_ASSET_CFGS[0]
    fixed_bolt_m8 = _FIXED_ASSET_CFGS[1]
    fixed_bolt_m12 = _FIXED_ASSET_CFGS[2]
    fixed_bolt_m16 = _FIXED_ASSET_CFGS[3]
    fixed_gear_base = _FIXED_ASSET_CFGS[4]
    fixed_round_hole_4mm = _FIXED_ASSET_CFGS[5]
    fixed_round_hole_8mm = _FIXED_ASSET_CFGS[6]
    fixed_round_hole_12mm = _FIXED_ASSET_CFGS[7]
    fixed_round_hole_16mm = _FIXED_ASSET_CFGS[8]
    fixed_rectangular_hole_4mm = _FIXED_ASSET_CFGS[9]
    fixed_rectangular_hole_8mm = _FIXED_ASSET_CFGS[10]
    fixed_rectangular_hole_12mm = _FIXED_ASSET_CFGS[11]
    fixed_rectangular_hole_16mm = _FIXED_ASSET_CFGS[12]
    fixed_usb_a_socket = _FIXED_ASSET_CFGS[13]
    fixed_waterproof_socket = _FIXED_ASSET_CFGS[14]
    fixed_bnc_socket = _FIXED_ASSET_CFGS[15]
    fixed_dsub_socket = _FIXED_ASSET_CFGS[16]
    fixed_rj45_socket = _FIXED_ASSET_CFGS[17]

    held_00 = _HELD_ASSET_CFGS[0]
    held_01 = _HELD_ASSET_CFGS[1]
    held_02 = _HELD_ASSET_CFGS[2]
    held_03 = _HELD_ASSET_CFGS[3]
    held_04 = _HELD_ASSET_CFGS[4]
    held_05 = _HELD_ASSET_CFGS[5]
    held_06 = _HELD_ASSET_CFGS[6]
    held_07 = _HELD_ASSET_CFGS[7]
    held_08 = _HELD_ASSET_CFGS[8]
    held_09 = _HELD_ASSET_CFGS[9]
    held_10 = _HELD_ASSET_CFGS[10]
    held_11 = _HELD_ASSET_CFGS[11]
    held_12 = _HELD_ASSET_CFGS[12]
    held_13 = _HELD_ASSET_CFGS[13]
    held_14 = _HELD_ASSET_CFGS[14]
    held_15 = _HELD_ASSET_CFGS[15]
    held_16 = _HELD_ASSET_CFGS[16]
    held_17 = _HELD_ASSET_CFGS[17]
    held_18 = _HELD_ASSET_CFGS[18]
    held_19 = _HELD_ASSET_CFGS[19]

    dome_light = assets.DOMELIGHT_CFG
