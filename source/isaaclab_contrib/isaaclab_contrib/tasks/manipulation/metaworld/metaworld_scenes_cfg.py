# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named scene cfgs for every Meta-World+ task family.

Inspired by ``factory_v1/factory_scenes_cfg.py``: instead of generating
scenes through a ``_build_scene_class(asset_cfg, source, marker)`` factory
hidden inside ``env_cfgs.py``, each task family gets a public, grep-able
scene class:

* :class:`SawyerCubeSceneCfg`        — MT3 reach/push/pick-place (cube IS the manipulandum).
* :class:`SawyerDrawerSceneCfg`      — drawer-open / drawer-close.
* :class:`SawyerButtonSceneCfg`      — button-press-topdown / coffee-button.
* :class:`SawyerWindowSceneCfg`      — window-open / window-close.
* :class:`SawyerFaucetSceneCfg`      — faucet-open/close, dial-turn, lever-pull.
* :class:`SawyerDoorSceneCfg`        — door-open/close, door-lock, door-unlock.
* :class:`SawyerPegInsertSceneCfg`   — peg-insert-side (cube becomes a peg cylinder).

Inheritance:

* :class:`MetaworldSceneCfg` (in ``metaworld_env_base``) holds only the
  globals shared by every task — robot anchor, ``tcp_frame``, lighting,
  goal marker. Notably it has *no* ``cube`` field; that's per-family.
* :class:`SawyerCubeSceneCfg` is for MT3 (cube *is* the manipulandum,
  visible 4 cm cube).
* :class:`_SawyerArticulatedSceneCfg` is an intermediate base for the
  six real-asset families — Sawyer + 1 mm hidden anchor cube + tcp_frame.
  The 6 articulated scenes inherit from it and only set ``cabinet`` +
  ``keypoint_frame``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from .metaworld_assets_cfg import (
    MW_ASSEMBLY_PEG_CFG,
    MW_BASKET_CFG,
    MW_BIN_CFG,
    MW_BOX_WITH_LID_CFG,
    MW_BUTTON_CFG,
    MW_BUTTON_FRONT_CFG,
    MW_DOOR_CFG,
    MW_DRAWER_CFG,
    MW_FAUCET_CFG,
    MW_HANDLE_PRESS_CFG,
    MW_HANDLE_PRESS_SIDE_CFG,
    MW_HOLE_BLOCK_CFG,
    MW_NAIL_BLOCK_CFG,
    MW_PEG_BLOCK_CFG,
    MW_PEG_UNPLUG_CFG,
    MW_PLATE_CFG,
    MW_PLATE_SIDE_CFG,
    MW_SHELF_CFG,
    MW_SOCCER_GOAL_CFG,
    MW_STICK_CFG,
    MW_WALL_CFG,
    MW_WINDOW_CFG,
    SAWYER_METAWORLD_CFG,
)
from .metaworld_env_base import MetaworldSceneCfg

# ─── Shared scene primitives ──────────────────────────────────────────────


def _tcp_frame() -> FrameTransformerCfg:
    """Sawyer TCP frame transformer (mean of leftpad + rightpad).

    Tracks the *fingertip* of each pad — Meta-World's ``tcp_center`` is the
    mean of two MJCF *sites* (``leftEndEffector`` / ``rightEndEffector``)
    that sit 4.5 cm below the pad COMs (z=0.150 vs COM z=0.195 in MW's
    MJCF). Our USD doesn't have those sites, so we bake the same offset
    into the FrameTransformer so all reward atoms operate on the fingertip
    points natively.
    """
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/leftpad",
                name="leftpad",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.045)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/rightpad",
                name="rightpad",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.045)),
            ),
        ],
    )


def _asset_keypoint_frame(asset_prim: str, source_body: str, marker_body: str) -> FrameTransformerCfg:
    """Keypoint frame pointing at ``marker_body`` welded inside ``asset_prim``.

    Args:
        asset_prim: The asset's spawn prim name (typically ``"Cabinet"`` for
            real-asset scenes).
        source_body: Body whose transform is the reference frame for the
            transformer (typically the kinematic root, e.g. ``"drawercase"``).
        marker_body: The welded marker body to track (e.g. ``"drawer_handle"``).
    """
    return FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{asset_prim}/{source_body}",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{asset_prim}/{marker_body}",
                name="kp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )


def _cube_keypoint_frame() -> FrameTransformerCfg:
    """Keypoint frame pointing at the cube's root prim (MT3 manipulandum)."""
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Cube", name="kp"),
        ],
    )


# 4 cm visible cube — MT3 manipulandum.
MT3_VISIBLE_CUBE_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.04, 0.04, 0.04),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.75),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        # Match MW's MJCF cube friction ``[1, 0.1, 0.002]`` (slide, spin,
        # roll). PhysX defaults are lower, which means the pads slip on the
        # cube without imparting force — push agent caged the cube reward
        # but couldn't actually move it.
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, 0.02)),
)


# 1 mm invisible anchor cube — used by every non-MT3 articulated scene so that
# :class:`MetaworldPairedCommand` has a RigidObject to write the per-env goal
# pose to (the asset reward reads ``keypoint_frame``, not the cube).
HIDDEN_CUBE_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.001, 0.001, 0.001),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0), opacity=0.0),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, 0.02)),
)


# ─── Scene classes ────────────────────────────────────────────────────────


@configclass
class SawyerCubeSceneCfg(MetaworldSceneCfg):
    """MT3 scene — the cube is the manipulandum. Used by reach / push / pick-place."""

    robot = SAWYER_METAWORLD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = MT3_VISIBLE_CUBE_CFG
    tcp_frame: FrameTransformerCfg = _tcp_frame()
    keypoint_frame: FrameTransformerCfg = _cube_keypoint_frame()


@configclass
class _SawyerArticulatedSceneCfg(MetaworldSceneCfg):
    """Intermediate base for the 6 real-asset Sawyer scenes.

    Carries the Sawyer + tcp_frame + 1 mm hidden anchor cube. Subclasses
    only set ``cabinet`` (the articulated MW asset) and ``keypoint_frame``
    (the FrameTransformer pointing at the welded marker).
    """

    robot = SAWYER_METAWORLD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = HIDDEN_CUBE_CFG
    tcp_frame: FrameTransformerCfg = _tcp_frame()


@configclass
class SawyerDrawerSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + MW drawer at world (0, 0.9, 0.084). Used by drawer-open/close.

    Closed-drawer handle world position: ``(0, 0.74, 0.09)``.
    """

    cabinet = MW_DRAWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "drawercase", "drawer_handle")


@configclass
class SawyerButtonSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + MW button. Used by button-press-topdown and coffee-button."""

    cabinet = MW_BUTTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "button_box", "button_top")


@configclass
class SawyerWindowSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + MW window. Used by window-open and window-close."""

    cabinet = MW_WINDOW_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "window_frame", "window_handle")


@configclass
class SawyerFaucetSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + MW faucet. Used by faucet-open/close, dial-turn, lever-pull."""

    cabinet = MW_FAUCET_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "faucet_base", "handle_tip")


@configclass
class SawyerDoorSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + MW door. Used by door-open/close, door-lock, door-unlock."""

    cabinet = MW_DOOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "door_frame", "door_handle")


@configclass
class SawyerPegInsertSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + MW peg-block. Used by peg-insert-side.

    Unlike the other articulated scenes, ``cube`` is overridden to be a
    real peg cylinder (collidable, with mass) — the agent picks it up and
    inserts into the side hole.
    """

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CylinderCfg(
            radius=0.013,
            height=0.06,
            axis="X",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.65, 0.04)),
    )
    cabinet = MW_PEG_BLOCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "block", "hole")


# ─── Cube + auxiliary kinematic asset (visible cube IS the manipulandum) ────


@configclass
class SawyerCubeWithWallSceneCfg(SawyerCubeSceneCfg):
    """Cube + kinematic wall obstacle. Used by push-wall, reach-wall, pick-place-wall."""

    wall = MW_WALL_CFG.replace(prim_path="{ENV_REGEX_NS}/Wall")


@configclass
class SawyerBasketSceneCfg(SawyerCubeSceneCfg):
    """Cube (ball) + basketball hoop. Used by basketball."""

    basket = MW_BASKET_CFG.replace(prim_path="{ENV_REGEX_NS}/Basket")


@configclass
class SawyerShelfSceneCfg(SawyerCubeSceneCfg):
    """Cube + elevated shelf. Used by shelf-place."""

    shelf = MW_SHELF_CFG.replace(prim_path="{ENV_REGEX_NS}/Shelf")


@configclass
class SawyerSoccerSceneCfg(SawyerCubeSceneCfg):
    """Cube (ball) + soccer goal. Used by soccer."""

    soccer_goal = MW_SOCCER_GOAL_CFG.replace(prim_path="{ENV_REGEX_NS}/SoccerGoal")


@configclass
class SawyerCubeWithBinSceneCfg(SawyerCubeSceneCfg):
    """Cube + open-top bin. Used by sweep-into."""

    bin = MW_BIN_CFG.replace(prim_path="{ENV_REGEX_NS}/Bin")


# ─── Articulated plate (plate IS the manipulandum) ─────────────────────────


@configclass
class SawyerPlateSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + MW plate (slides in horizontal groove). Used by plate-slide × 4."""

    cabinet = MW_PLATE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "groove_base", "plate_marker")


# ─── Articulated handle / peg-unplug / box-with-lid (manipulandum is the asset) ───


@configclass
class SawyerHandlePressSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + top-down handle (prismatic Z). Used by handle-press / handle-pull."""

    cabinet = MW_HANDLE_PRESS_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "handle_base", "handle_top")


@configclass
class SawyerHandlePressSideSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + side-mounted handle (prismatic X). Used by handle-press-side / handle-pull-side."""

    cabinet = MW_HANDLE_PRESS_SIDE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "handle_base", "handle_tip")


@configclass
class SawyerPegUnplugSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + wall-mounted peg (prismatic X, pulled out). Used by peg-unplug-side."""

    cabinet = MW_PEG_UNPLUG_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "wall_base", "peg_tip")


@configclass
class SawyerBoxWithLidSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + box with hinged lid. Used by box-close."""

    cabinet = MW_BOX_WITH_LID_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "box_base", "lid_marker")


@configclass
class SawyerHoleBlockSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + kinematic block with a top pocket. Used by hand-insert (TCP target,
    keypoint at pocket marker)."""

    cabinet = MW_HOLE_BLOCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "block_base", "hole_marker")


@configclass
class SawyerCubeInHoleSceneCfg(SawyerHoleBlockSceneCfg):
    """Hole-block scene where the cube IS the manipulandum (cube starts in
    the pocket; agent lifts it out). Overrides the inherited ``cube`` (1 mm
    hidden anchor) with the visible 4 cm cube and points ``keypoint_frame``
    at the cube root rather than the pocket marker.

    Used by pick-out-of-hole.
    """

    cube: RigidObjectCfg = MT3_VISIBLE_CUBE_CFG
    keypoint_frame: FrameTransformerCfg = _cube_keypoint_frame()


# ─── Cube + button (existing button scene) + wall ───────────────────────────


@configclass
class SawyerButtonWithWallSceneCfg(SawyerButtonSceneCfg):
    """Sawyer + button + wall obstacle. Used by button-press-topdown-wall."""

    wall = MW_WALL_CFG.replace(prim_path="{ENV_REGEX_NS}/Wall")


# ─── Cube + decorative kinematic asset (cube IS the manipulandum) ──────────


@configclass
class SawyerCubeWithButtonSceneCfg(SawyerCubeSceneCfg):
    """Cube + kinematic button (coffee-machine decoration). Used by coffee-push, coffee-pull.

    The button is articulated but its joint isn't actuated by the agent;
    it sits as a fixture the agent pushes the cube toward/away from.
    """

    coffee_button = MW_BUTTON_CFG.replace(prim_path="{ENV_REGEX_NS}/CoffeeButton")


@configclass
class SawyerCubeWithStickSceneCfg(SawyerCubeSceneCfg):
    """Cube + free stick (tool). Used by stick-push, stick-pull.

    The stick is a free :class:`RigidObject` the agent could optionally
    grasp and use as a tool; for verification we drive the cube directly.
    """

    stick = MW_STICK_CFG


@configclass
class SawyerCubeWithPegSceneCfg(SawyerCubeSceneCfg):
    """Cube + kinematic peg-stand. Used by assembly (cube placed onto peg)
    and disassemble (cube starts on peg, lifted off)."""

    assembly_peg = MW_ASSEMBLY_PEG_CFG.replace(prim_path="{ENV_REGEX_NS}/AssemblyPeg")


# ─── Articulated nail (nail IS the manipulandum) ────────────────────────────


@configclass
class SawyerNailBlockSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + nail-block (articulated; nail slides into block when struck).
    Used by hammer."""

    cabinet = MW_NAIL_BLOCK_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "block_base", "nail_head")


# ─── Plate-side and front-button scenes (require dedicated USDs) ───────────


@configclass
class SawyerPlateSideSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + MW plate sliding along world Y. Used by plate-slide-side and
    plate-slide-back-side. Distinct from :class:`SawyerPlateSceneCfg` which
    uses the X-axis plate USD."""

    cabinet = MW_PLATE_SIDE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "groove_base", "plate_marker")


@configclass
class SawyerButtonFrontSceneCfg(_SawyerArticulatedSceneCfg):
    """Sawyer + front-facing button. Used by button-press."""

    cabinet = MW_BUTTON_FRONT_CFG.replace(prim_path="{ENV_REGEX_NS}/Cabinet")
    keypoint_frame: FrameTransformerCfg = _asset_keypoint_frame("Cabinet", "button_box", "button_top")


@configclass
class SawyerButtonFrontWithWallSceneCfg(SawyerButtonFrontSceneCfg):
    """Sawyer + front-facing button + wall obstacle. Used by button-press-wall."""

    wall = MW_WALL_CFG.replace(prim_path="{ENV_REGEX_NS}/Wall")


__all__ = [
    "MT3_VISIBLE_CUBE_CFG",
    "HIDDEN_CUBE_CFG",
    "SawyerCubeSceneCfg",
    "SawyerDrawerSceneCfg",
    "SawyerButtonSceneCfg",
    "SawyerWindowSceneCfg",
    "SawyerFaucetSceneCfg",
    "SawyerDoorSceneCfg",
    "SawyerPegInsertSceneCfg",
    # Tier-2 scenes (cube + auxiliary asset; visible cube is the manipulandum)
    "SawyerCubeWithWallSceneCfg",
    "SawyerBasketSceneCfg",
    "SawyerShelfSceneCfg",
    "SawyerSoccerSceneCfg",
    "SawyerCubeWithBinSceneCfg",
    # Tier-2 articulated plate
    "SawyerPlateSceneCfg",
    # Tier-2 handle / peg-unplug / box / hole-block / button-wall
    "SawyerHandlePressSceneCfg",
    "SawyerHandlePressSideSceneCfg",
    "SawyerPegUnplugSceneCfg",
    "SawyerBoxWithLidSceneCfg",
    "SawyerHoleBlockSceneCfg",
    "SawyerCubeInHoleSceneCfg",
    "SawyerButtonWithWallSceneCfg",
    # Tier-3 cube + decorative / dual-object scenes
    "SawyerCubeWithButtonSceneCfg",
    "SawyerCubeWithStickSceneCfg",
    "SawyerCubeWithPegSceneCfg",
    "SawyerNailBlockSceneCfg",
    # Tier-4 plate-side / front-button (dedicated USDs)
    "SawyerPlateSideSceneCfg",
    "SawyerButtonFrontSceneCfg",
    "SawyerButtonFrontWithWallSceneCfg",
]
