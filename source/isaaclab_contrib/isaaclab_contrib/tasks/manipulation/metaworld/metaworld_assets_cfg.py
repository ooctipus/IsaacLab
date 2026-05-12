# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""All Meta-World+ asset cfgs in one module.

Inspired by ``factory_v1/factory_assets_cfg.py``: every asset cfg lives here
so ``grep`` finds it in one place. The actual USD files stay under
``assets/<asset>/usd/mw_<asset>.usda`` (real files on disk that the cfgs
reference); this module just builds the wrapping :class:`ArticulationCfg`
or :class:`RigidObjectCfg`.

Asset inventory (24 cfgs covering the Sawyer + all 50 MT50 manipulanda):

* **Robot** (1): :data:`SAWYER_METAWORLD_CFG`.
* **Articulated MW manipulanda** (19) — original 6 + 13 added for MT50.
* **Free rigid objects** (4) — assembly ring, hammer, mug, stick (primitive
  shapes, no USD).

USD-authoring helpers and builders live in ``utils/usd/`` (mirrors
``factory_v1``'s split between ``assets/`` (data) and ``utils/`` (tools)).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg

_HERE = os.path.dirname(__file__)


def _usd(asset_name: str, file_name: str | None = None) -> str:
    """Resolve ``assets/<asset_name>/usd/<file>`` (default: ``mw_<asset>.usda``)."""
    return os.path.join(_HERE, "assets", asset_name, "usd", file_name or f"mw_{asset_name}.usda")


# ─── Common physics / articulation defaults ────────────────────────────────

_RIGID_DEFAULTS = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0)
_ART_DEFAULTS = sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)
_ART_DEFAULTS_HIPREC = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=0,
)
_RIGID_FRICTION = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)


# ===========================================================================
# Robot — Sawyer arm + grafted Meta-World box gripper
# ===========================================================================

SAWYER_USD_PATH: str = _usd("sawyer", "sawyer_with_gripper.usda")
"""Path to the pre-built Sawyer-with-gripper USD on disk."""

SAWYER_METAWORLD_CFG: ArticulationCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SAWYER_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Disable gravity to mirror Meta-World's mocap-welded "floating arm"
            # behaviour. Without it the arm sags and DiffIK has to constantly
            # fight gravity. Same trick as FRANKA_PANDA_HIGH_PD_CFG.
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=_ART_DEFAULTS_HIPREC,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Read off Meta-World's MuJoCo state after ``_reset_hand`` settles
            # (i.e. what the mocap weld drags the arm to when targeting
            # ``hand_init_pos = (0, 0.6, 0.2)``). MW's MJCF places the Sawyer
            # base at world origin (matches our ``init_state.pos``), so these
            # joint defaults land the pad COMs near MW's
            # ``(0.005, 0.60, 0.195)`` and align caging margins exactly.
            "right_j0": 1.886,
            "right_j1": -0.581,
            "right_j2": -0.971,
            "right_j3": 1.642,
            "right_j4": 0.935,
            "right_j5": 1.041,
            "right_j6": 2.291,
            "r_close": 0.0,
            "l_close": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["right_j[0-6]"],
            effort_limit_sim={
                "right_j[0-1]": 80.0,
                "right_j[2-3]": 40.0,
                "right_j[4-6]": 9.0,
            },
            # High-PD gains so DiffIK targets are tracked stably (matches
            # FRANKA_PANDA_HIGH_PD_CFG's 400/80 with disable_gravity=True).
            stiffness=400.0,
            damping=80.0,
        ),
        "head": ImplicitActuatorCfg(joint_names_expr=["head_pan"], effort_limit_sim=8.0, stiffness=80.0, damping=4.0),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["[rl]_close"], effort_limit_sim=200.0, stiffness=400.0, damping=10.0
        ),
    },
)
"""Sawyer arm + parallel-jaw gripper grafted onto Nucleus instanceable Sawyer."""


# ===========================================================================
# MW manipulanda — original 6 (single-joint articulated assets)
# ===========================================================================


def _articulated(
    name: str,
    *,
    joint_name: str,
    actuator_name: str,
    damping: float = 1.0,
    stiffness: float = 0.0,
    activate_contact_sensors: bool = True,
    init_pos: float = 0.0,
) -> ArticulationCfg:
    """Helper: build a single-joint MW articulation cfg from an asset name."""
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_usd(name),
            rigid_props=_RIGID_DEFAULTS,
            articulation_props=_ART_DEFAULTS,
            activate_contact_sensors=activate_contact_sensors,
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={joint_name: init_pos}),
        actuators={
            actuator_name: ImplicitActuatorCfg(
                joint_names_expr=[joint_name], effort_limit_sim=200.0, stiffness=stiffness, damping=damping
            ),
        },
    )


def _kinematic(name: str, *, activate_contact_sensors: bool = False) -> ArticulationCfg:
    """Helper: build a kinematic MW articulation cfg (no actuated joints).

    Used for assets welded to world via a fixed joint inside the USD —
    walls, bins, shelves, hoops, peg-stands, hole-blocks, etc. The empty
    ``joint_pos``/``joint_vel`` dicts override the default ``{".*": 0.0}``
    regex, which would error on assets that genuinely have zero joints.
    """
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=_usd(name),
            rigid_props=_RIGID_DEFAULTS,
            articulation_props=_ART_DEFAULTS,
            activate_contact_sensors=activate_contact_sensors,
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={}, joint_vel={}),
        actuators={},
    )


MW_DRAWER_USD_PATH: str = _usd("drawer")
"""Slider drawer (prismatic Y, range ``[-0.16, 0]``)."""
MW_DRAWER_CFG: ArticulationCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=MW_DRAWER_USD_PATH,
        rigid_props=_RIGID_DEFAULTS,
        articulation_props=_ART_DEFAULTS_HIPREC,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"goal_slidey": 0.0}),
    actuators={
        # Soft drive matching MW's MJCF damping=2.
        "drawer": ImplicitActuatorCfg(
            joint_names_expr=["goal_slidey"], effort_limit_sim=200.0, stiffness=0.0, damping=2.0
        ),
    },
)


MW_BUTTON_USD_PATH: str = _usd("button")
"""Push button (prismatic Z, range ``[-0.06, 0]``)."""
MW_BUTTON_CFG: ArticulationCfg = _articulated(
    "button", joint_name="btnbox_joint", actuator_name="button", stiffness=0.5, damping=1.0
)


MW_WINDOW_USD_PATH: str = _usd("window")
"""Sliding window (prismatic X, range ``[0, 0.20]``)."""
MW_WINDOW_CFG: ArticulationCfg = _articulated("window", joint_name="window_slide", actuator_name="window", damping=1.0)


MW_FAUCET_USD_PATH: str = _usd("faucet")
"""Faucet handle (revolute Z, range ``[-π/2, π/2]``)."""
MW_FAUCET_CFG: ArticulationCfg = _articulated("faucet", joint_name="knob_Joint_1", actuator_name="faucet", damping=2.0)


MW_DOOR_USD_PATH: str = _usd("door")
"""Door panel (revolute Z, range ``[-π/2, 0]``)."""
MW_DOOR_CFG: ArticulationCfg = _articulated("door", joint_name="door_hinge", actuator_name="door", damping=2.0)


MW_PEG_BLOCK_USD_PATH: str = _usd("peg", "mw_peg_block.usda")
"""Perforated wall (kinematic; the peg is a separate :class:`RigidObjectCfg`)."""
MW_PEG_BLOCK_CFG: ArticulationCfg = _kinematic("peg", activate_contact_sensors=False)
# Override USD path (asset folder is "peg" but USD is mw_peg_block.usda).
MW_PEG_BLOCK_CFG.spawn.usd_path = MW_PEG_BLOCK_USD_PATH


# ===========================================================================
# MW manipulanda — new 13 (MT50 expansion)
# ===========================================================================


MW_WALL_USD_PATH: str = _usd("wall")
"""Kinematic wall obstacle (push-wall, reach-wall, button-press-wall, …)."""
MW_WALL_CFG: ArticulationCfg = _kinematic("wall")


MW_HANDLE_PRESS_USD_PATH: str = _usd("handle_press")
"""Top-down handle (prismatic Z, range ``[-0.10, +0.10]``)."""
MW_HANDLE_PRESS_CFG: ArticulationCfg = _articulated(
    "handle_press", joint_name="handle_slide", actuator_name="handle", damping=1.0
)


MW_HANDLE_PRESS_SIDE_USD_PATH: str = _usd("handle_press_side")
"""Side-mounted handle (prismatic X, range ``[-0.10, +0.10]``)."""
MW_HANDLE_PRESS_SIDE_CFG: ArticulationCfg = _articulated(
    "handle_press_side", joint_name="handle_slide", actuator_name="handle", damping=1.0
)


MW_PLATE_USD_PATH: str = _usd("plate")
"""Sliding plate (prismatic X, range ``[-0.20, +0.20]``)."""
MW_PLATE_CFG: ArticulationCfg = _articulated("plate", joint_name="plate_slide", actuator_name="plate", damping=0.5)


MW_BASKET_USD_PATH: str = _usd("basket")
"""Basketball hoop (kinematic)."""
MW_BASKET_CFG: ArticulationCfg = _kinematic("basket")


MW_BIN_USD_PATH: str = _usd("bin")
"""Open-top bin (kinematic; bin-picking, sweep-into)."""
MW_BIN_CFG: ArticulationCfg = _kinematic("bin")


MW_ASSEMBLY_PEG_USD_PATH: str = _usd("assembly_peg")
"""Vertical peg-stand (kinematic; assembly, disassemble)."""
MW_ASSEMBLY_PEG_CFG: ArticulationCfg = _kinematic("assembly_peg")


MW_BOX_WITH_LID_USD_PATH: str = _usd("box_with_lid")
"""Box with a hinged lid (revolute X, range ``[0°, 120°]``)."""
MW_BOX_WITH_LID_CFG: ArticulationCfg = _articulated(
    "box_with_lid", joint_name="lid_hinge", actuator_name="lid", damping=2.0
)


MW_SHELF_USD_PATH: str = _usd("shelf")
"""Elevated shelf (kinematic; shelf-place)."""
MW_SHELF_CFG: ArticulationCfg = _kinematic("shelf")


MW_SOCCER_GOAL_USD_PATH: str = _usd("soccer_goal")
"""Open-front soccer goal (kinematic)."""
MW_SOCCER_GOAL_CFG: ArticulationCfg = _kinematic("soccer_goal")


MW_NAIL_BLOCK_USD_PATH: str = _usd("nail_block")
"""Block + nail (prismatic Z, range ``[-0.06, 0]``)."""
MW_NAIL_BLOCK_CFG: ArticulationCfg = _articulated(
    "nail_block", joint_name="nail_drive", actuator_name="nail", damping=2.0
)


MW_HOLE_BLOCK_USD_PATH: str = _usd("hole_block")
"""Block with a top pocket (kinematic; pick-out-of-hole, hand-insert)."""
MW_HOLE_BLOCK_CFG: ArticulationCfg = _kinematic("hole_block")


MW_PEG_UNPLUG_USD_PATH: str = _usd("peg_unplug")
"""Wall-mounted peg (prismatic X, range ``[-0.10, 0]``)."""
MW_PEG_UNPLUG_CFG: ArticulationCfg = _articulated(
    "peg_unplug", joint_name="peg_slide", actuator_name="peg", damping=1.0
)


MW_PLATE_SIDE_USD_PATH: str = _usd("plate_side")
"""Plate sliding along world Y (prismatic Y, range ``[-0.20, +0.20]``).

Same shape as :data:`MW_PLATE_CFG` but with the joint axis rotated 90°
so the plate moves laterally instead of forward/back. Used by
plate-slide-side and plate-slide-back-side.
"""
MW_PLATE_SIDE_CFG: ArticulationCfg = _articulated(
    "plate_side", joint_name="plate_slide", actuator_name="plate", damping=0.5
)


MW_BUTTON_FRONT_USD_PATH: str = _usd("button_front")
"""Front-facing button on a wall mount (prismatic Y, range ``[0, +0.06]``).

Cap protrudes toward the agent at ``joint=0`` and is pressed +Y into the
mount at ``joint=+0.06``. Used by button-press and button-press-wall (vs
the topdown button which slides in -Z).
"""
MW_BUTTON_FRONT_CFG: ArticulationCfg = _articulated(
    "button_front", joint_name="btnbox_joint", actuator_name="button", stiffness=0.5, damping=1.0
)


# ===========================================================================
# Free :class:`RigidObjectCfg` (no USD; primitive shapes)
# ===========================================================================


MW_ASSEMBLY_RING_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/AssemblyRing",
    spawn=sim_utils.CylinderCfg(
        radius=0.03,
        height=0.015,
        axis="Z",
        rigid_props=_RIGID_DEFAULTS,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=_RIGID_FRICTION,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, 0.04)),
)
"""Assembly ring approximated as a 6 cm × 1.5 cm flat cylinder. PhysX rigid-body
collisions don't support concave geometry, so the agent gripper pinches the
side rather than grasping through a hole."""


MW_HAMMER_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Hammer",
    spawn=sim_utils.CuboidCfg(
        size=(0.04, 0.16, 0.025),
        rigid_props=_RIGID_DEFAULTS,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.20),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=_RIGID_FRICTION,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.20)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, 0.04)),
)
"""Hammer approximated as a 4 × 16 × 2.5 cm cuboid (handle + head fused).
Mass 0.20 kg matches MW."""


MW_MUG_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Mug",
    spawn=sim_utils.CylinderCfg(
        radius=0.025,
        height=0.06,
        axis="Z",
        rigid_props=_RIGID_DEFAULTS,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=_RIGID_FRICTION,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.95)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, 0.03)),
)
"""Coffee mug approximated as a 5 cm × 6 cm cylinder."""


MW_STICK_CFG: RigidObjectCfg = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Stick",
    spawn=sim_utils.CylinderCfg(
        radius=0.012,
        height=0.20,
        axis="X",
        rigid_props=_RIGID_DEFAULTS,
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=_RIGID_FRICTION,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.5, 0.3)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.6, 0.03)),
)
"""Stick: 1.2 cm radius × 20 cm long horizontal cylinder."""


__all__ = [
    # Robot
    "SAWYER_METAWORLD_CFG",
    "SAWYER_USD_PATH",
    # Articulated MW manipulanda — original 6
    "MW_DRAWER_CFG",
    "MW_DRAWER_USD_PATH",
    "MW_BUTTON_CFG",
    "MW_BUTTON_USD_PATH",
    "MW_WINDOW_CFG",
    "MW_WINDOW_USD_PATH",
    "MW_FAUCET_CFG",
    "MW_FAUCET_USD_PATH",
    "MW_DOOR_CFG",
    "MW_DOOR_USD_PATH",
    "MW_PEG_BLOCK_CFG",
    "MW_PEG_BLOCK_USD_PATH",
    # Articulated MW manipulanda — new 13 (MT50 expansion)
    "MW_WALL_CFG",
    "MW_WALL_USD_PATH",
    "MW_HANDLE_PRESS_CFG",
    "MW_HANDLE_PRESS_USD_PATH",
    "MW_HANDLE_PRESS_SIDE_CFG",
    "MW_HANDLE_PRESS_SIDE_USD_PATH",
    "MW_PLATE_CFG",
    "MW_PLATE_USD_PATH",
    "MW_BASKET_CFG",
    "MW_BASKET_USD_PATH",
    "MW_BIN_CFG",
    "MW_BIN_USD_PATH",
    "MW_ASSEMBLY_PEG_CFG",
    "MW_ASSEMBLY_PEG_USD_PATH",
    "MW_BOX_WITH_LID_CFG",
    "MW_BOX_WITH_LID_USD_PATH",
    "MW_SHELF_CFG",
    "MW_SHELF_USD_PATH",
    "MW_SOCCER_GOAL_CFG",
    "MW_SOCCER_GOAL_USD_PATH",
    "MW_NAIL_BLOCK_CFG",
    "MW_NAIL_BLOCK_USD_PATH",
    "MW_HOLE_BLOCK_CFG",
    "MW_HOLE_BLOCK_USD_PATH",
    "MW_PEG_UNPLUG_CFG",
    "MW_PEG_UNPLUG_USD_PATH",
    "MW_PLATE_SIDE_CFG",
    "MW_PLATE_SIDE_USD_PATH",
    "MW_BUTTON_FRONT_CFG",
    "MW_BUTTON_FRONT_USD_PATH",
    # Free RigidObjectCfgs (no USD)
    "MW_ASSEMBLY_RING_CFG",
    "MW_HAMMER_CFG",
    "MW_MUG_CFG",
    "MW_STICK_CFG",
]
