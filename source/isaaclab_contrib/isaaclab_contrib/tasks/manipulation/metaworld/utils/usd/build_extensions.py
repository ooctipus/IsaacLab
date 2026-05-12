# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author the remaining 13 MW-style task assets to cover MT50.

Outputs (each ``<dir>/usd/mw_<name>.usda``):

* ``wall``              — kinematic wall obstacle (push-wall, reach-wall, …).
* ``handle_press``      — vertical post + handle on prismatic Z (handle-press, handle-pull).
* ``handle_press_side`` — vertical post + handle on prismatic X (handle-press-side, handle-pull-side).
* ``plate``             — sliding plate in a groove (plate-slide × 4).
* ``basket``            — basketball hoop on a base (kinematic).
* ``bin``               — open-top container (bin-picking, sweep-into).
* ``assembly_peg``      — kinematic peg-stand for ring placement (assembly, disassemble).
* ``box_with_lid``      — box base + hinged lid revolute (box-close).
* ``shelf``             — kinematic elevated platform (shelf-place).
* ``soccer_goal``       — kinematic open-front goal (soccer).
* ``nail_block``        — block + nail on prismatic Z (hammer).
* ``hole_block``        — kinematic block with cube-shaped hole (pick-out-of-hole, hand-insert).
* ``peg_unplug``        — block with horizontal peg on prismatic X (peg-unplug-side).

Each USD follows the same flat-sibling pattern as the existing 5 MW USDs.
Geometry is simplified to primitive boxes + cylinders matching the
relevant surfaces, joint axes/ranges, and marker positions needed for
reward terms and scripted-policy verification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# This script lives at utils/usd/; the per-asset USDs go to assets/<name>/usd/.
_ASSETS_ROOT = Path(__file__).resolve().parents[2] / "assets"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out_root", default=str(_ASSETS_ROOT))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

simulation_app = AppLauncher(args).app

from pxr import UsdPhysics  # noqa: E402

from isaaclab_contrib.tasks.manipulation.metaworld.utils.usd.helpers import (  # noqa: E402
    add_box_geom,
    add_cylinder_geom,
    add_fixed_joint,
    add_fixed_joint_to_world,
    add_handle_marker,
    add_prismatic_joint,
    add_revolute_joint,
    add_rigid_body_anchor,
    stage_init,
)

OUT_ROOT = Path(args.out_root).resolve()


def _save(stage, rel_path: str) -> None:
    out = OUT_ROOT / rel_path
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage.GetRootLayer().Export(str(out))
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Wall (kinematic obstacle for push-wall / reach-wall / button-press-wall …)
# ============================================================================


def build_wall() -> None:
    """A simple wall obstacle. Kinematic, welded to world. Default world pose
    is set by the env-cfg (different tasks place the wall differently)."""
    out = OUT_ROOT / "wall" / "usd" / "mw_wall.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/wall", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.75, 0.06)
    add_rigid_body_anchor(stage, "/wall/wall_body", translate=base_pos, mass=10.0)
    # Slim vertical wall: 24 cm wide, 1 cm thick, 12 cm tall.
    add_box_geom(
        stage,
        "/wall/wall_body",
        "panel",
        half_extents=(0.12, 0.005, 0.06),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.5, 0.5, 0.5, 1.0),
    )
    add_handle_marker(stage, "/wall/wall_marker", world_pos=base_pos)
    add_fixed_joint(
        stage,
        "/wall/marker_to_wall",
        body0_path="/wall/wall_body",
        body1_path="/wall/wall_marker",
        local_pos0=(0.0, 0.0, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_fixed_joint_to_world(stage, "/wall/wall_to_world", body1_path="/wall/wall_body", world_pos=base_pos)
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Handle-Press (top-down): vertical post + handle on prismatic Z
# ============================================================================


def build_handle_press() -> None:
    """Top-down handle: a horizontal handle-bar on a vertical post that
    slides down (-Z) when pressed. Used by handle-press (push down) and
    handle-pull (lift up). Joint range covers both: [-0.10, 0.10]."""
    out = OUT_ROOT / "handle_press" / "usd" / "mw_handle_press.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/handle_press", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.85, 0.05)
    add_rigid_body_anchor(stage, "/handle_press/handle_base", translate=base_pos, mass=2.0)
    add_box_geom(
        stage,
        "/handle_press/handle_base",
        "stand",
        half_extents=(0.04, 0.04, 0.05),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.4, 0.4, 0.45, 1.0),
    )
    add_fixed_joint_to_world(
        stage, "/handle_press/base_to_world", body1_path="/handle_press/handle_base", world_pos=base_pos
    )

    # Sliding shaft + horizontal handle bar.
    handle_pos = (0.0, 0.85, 0.16)
    add_rigid_body_anchor(stage, "/handle_press/handle_link", translate=handle_pos, mass=0.05)
    add_box_geom(
        stage,
        "/handle_press/handle_link",
        "shaft",
        half_extents=(0.005, 0.005, 0.04),
        local_pos=(0.0, 0.0, -0.04),
        rgba=(0.8, 0.8, 0.8, 1.0),
    )
    add_box_geom(
        stage,
        "/handle_press/handle_link",
        "bar",
        half_extents=(0.04, 0.005, 0.005),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.85, 0.2, 0.2, 1.0),
    )
    add_handle_marker(stage, "/handle_press/handle_top", world_pos=handle_pos)
    add_fixed_joint(
        stage,
        "/handle_press/top_to_link",
        body0_path="/handle_press/handle_link",
        body1_path="/handle_press/handle_top",
        local_pos0=(0.0, 0.0, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    # Prismatic in Z, full range [-0.10, +0.10] so handle-press (down) and
    # handle-pull (up) both fit. Reset event picks the side.
    add_prismatic_joint(
        stage,
        "/handle_press/handle_slide",
        body0_path="/handle_press/handle_base",
        body1_path="/handle_press/handle_link",
        local_pos0=(0.0, 0.0, 0.11),
        local_pos1=(0.0, 0.0, 0.0),
        axis="Z",
        lower=-0.10,
        upper=0.10,
        damping=1.0,
        stiffness=0.0,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Handle-Press (side): vertical post + handle on prismatic X
# ============================================================================


def build_handle_press_side() -> None:
    """Side-mounted handle: handle slides along world X. Used by
    handle-press-side and handle-pull-side."""
    out = OUT_ROOT / "handle_press_side" / "usd" / "mw_handle_press_side.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/handle_press_side", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.85, 0.10)
    add_rigid_body_anchor(stage, "/handle_press_side/handle_base", translate=base_pos, mass=2.0)
    add_box_geom(
        stage,
        "/handle_press_side/handle_base",
        "stand",
        half_extents=(0.04, 0.04, 0.10),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.4, 0.4, 0.45, 1.0),
    )
    add_fixed_joint_to_world(
        stage, "/handle_press_side/base_to_world", body1_path="/handle_press_side/handle_base", world_pos=base_pos
    )

    # Handle that protrudes sideways (+X) and slides along X.
    handle_pos = (0.10, 0.85, 0.10)
    add_rigid_body_anchor(stage, "/handle_press_side/handle_link", translate=handle_pos, mass=0.05)
    add_box_geom(
        stage,
        "/handle_press_side/handle_link",
        "shaft",
        half_extents=(0.04, 0.005, 0.005),
        local_pos=(-0.04, 0.0, 0.0),
        rgba=(0.8, 0.8, 0.8, 1.0),
    )
    add_box_geom(
        stage,
        "/handle_press_side/handle_link",
        "bar",
        half_extents=(0.005, 0.005, 0.04),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.85, 0.2, 0.2, 1.0),
    )
    add_handle_marker(stage, "/handle_press_side/handle_tip", world_pos=handle_pos)
    add_fixed_joint(
        stage,
        "/handle_press_side/tip_to_link",
        body0_path="/handle_press_side/handle_link",
        body1_path="/handle_press_side/handle_tip",
        local_pos0=(0.0, 0.0, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_prismatic_joint(
        stage,
        "/handle_press_side/handle_slide",
        body0_path="/handle_press_side/handle_base",
        body1_path="/handle_press_side/handle_link",
        local_pos0=(0.04, 0.0, 0.0),
        local_pos1=(-0.06, 0.0, 0.0),
        axis="X",
        lower=-0.10,
        upper=0.10,
        damping=1.0,
        stiffness=0.0,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Plate (slides in a groove; prismatic X)
# ============================================================================


def build_plate() -> None:
    """Flat plate on a horizontal prismatic joint. Same asset for plate-slide
    (forward), plate-slide-back (reverse), plate-slide-side (lateral —
    achieved by spawning the asset rotated 90° in the env). The plate slides
    in a groove with railing walls."""
    out = OUT_ROOT / "plate" / "usd" / "mw_plate.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/plate", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.75, 0.02)
    add_rigid_body_anchor(stage, "/plate/groove_base", translate=base_pos, mass=2.0)
    # Groove: bottom + 2 side rails along world X.
    add_box_geom(
        stage,
        "/plate/groove_base",
        "bottom",
        half_extents=(0.20, 0.06, 0.005),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.45, 0.5, 0.55, 1.0),
    )
    add_box_geom(
        stage,
        "/plate/groove_base",
        "rail_l",
        half_extents=(0.20, 0.005, 0.015),
        local_pos=(0.0, -0.06, 0.015),
        rgba=(0.45, 0.5, 0.55, 1.0),
    )
    add_box_geom(
        stage,
        "/plate/groove_base",
        "rail_r",
        half_extents=(0.20, 0.005, 0.015),
        local_pos=(0.0, +0.06, 0.015),
        rgba=(0.45, 0.5, 0.55, 1.0),
    )
    add_fixed_joint_to_world(stage, "/plate/base_to_world", body1_path="/plate/groove_base", world_pos=base_pos)

    plate_pos = (-0.10, 0.75, 0.025)
    add_rigid_body_anchor(stage, "/plate/plate_link", translate=plate_pos, mass=0.10)
    add_box_geom(
        stage,
        "/plate/plate_link",
        "puck",
        half_extents=(0.04, 0.04, 0.01),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.85, 0.55, 0.20, 1.0),
    )
    add_handle_marker(stage, "/plate/plate_marker", world_pos=plate_pos)
    add_fixed_joint(
        stage,
        "/plate/marker_to_link",
        body0_path="/plate/plate_link",
        body1_path="/plate/plate_marker",
        local_pos0=(0.0, 0.0, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_prismatic_joint(
        stage,
        "/plate/plate_slide",
        body0_path="/plate/groove_base",
        body1_path="/plate/plate_link",
        local_pos0=(-0.10, 0.0, 0.005),
        local_pos1=(0.0, 0.0, 0.0),
        axis="X",
        lower=-0.20,
        upper=0.20,
        damping=0.5,
        stiffness=0.0,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Basket (basketball hoop, kinematic)
# ============================================================================


def build_basket() -> None:
    """Basketball hoop. Kinematic ring on a backboard. The "ball" is a free
    cube/ball spawned by the env; success = ball reaches the hoop center."""
    out = OUT_ROOT / "basket" / "usd" / "mw_basket.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/basket", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.85, 0.05)
    add_rigid_body_anchor(stage, "/basket/basket_base", translate=base_pos, mass=2.0)
    # backboard
    add_box_geom(
        stage,
        "/basket/basket_base",
        "post",
        half_extents=(0.005, 0.02, 0.10),
        local_pos=(0.0, 0.04, 0.10),
        rgba=(0.4, 0.4, 0.4, 1.0),
    )
    add_box_geom(
        stage,
        "/basket/basket_base",
        "board",
        half_extents=(0.08, 0.005, 0.08),
        local_pos=(0.0, 0.06, 0.20),
        rgba=(0.85, 0.85, 0.85, 1.0),
    )
    # rim — represented as 4 boxes forming an open square (5 cm radius)
    rim_z = 0.20
    add_box_geom(
        stage,
        "/basket/basket_base",
        "rim_n",
        half_extents=(0.06, 0.005, 0.005),
        local_pos=(0.0, -0.05, rim_z),
        rgba=(0.95, 0.4, 0.1, 1.0),
    )
    add_box_geom(
        stage,
        "/basket/basket_base",
        "rim_s",
        half_extents=(0.06, 0.005, 0.005),
        local_pos=(0.0, 0.05, rim_z),
        rgba=(0.95, 0.4, 0.1, 1.0),
    )
    add_box_geom(
        stage,
        "/basket/basket_base",
        "rim_e",
        half_extents=(0.005, 0.05, 0.005),
        local_pos=(0.06, 0.0, rim_z),
        rgba=(0.95, 0.4, 0.1, 1.0),
    )
    add_box_geom(
        stage,
        "/basket/basket_base",
        "rim_w",
        half_extents=(0.005, 0.05, 0.005),
        local_pos=(-0.06, 0.0, rim_z),
        rgba=(0.95, 0.4, 0.1, 1.0),
    )
    add_handle_marker(stage, "/basket/hoop_center", world_pos=(base_pos[0], base_pos[1], base_pos[2] + rim_z))
    add_fixed_joint(
        stage,
        "/basket/center_to_base",
        body0_path="/basket/basket_base",
        body1_path="/basket/hoop_center",
        local_pos0=(0.0, 0.0, rim_z),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_fixed_joint_to_world(stage, "/basket/base_to_world", body1_path="/basket/basket_base", world_pos=base_pos)
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Bin (open-top container; kinematic)
# ============================================================================


def build_bin() -> None:
    """Open-top container. Used by bin-picking (pick from one bin, place in
    another) and sweep-into (cube swept into bin). The bin's base is sized so
    a 4 cm cube fits with margin."""
    out = OUT_ROOT / "bin" / "usd" / "mw_bin.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/bin", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.75, 0.025)
    add_rigid_body_anchor(stage, "/bin/bin_base", translate=base_pos, mass=2.0)
    # Bottom + 4 walls.
    add_box_geom(
        stage,
        "/bin/bin_base",
        "bottom",
        half_extents=(0.06, 0.06, 0.005),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.5, 0.4, 0.3, 1.0),
    )
    add_box_geom(
        stage,
        "/bin/bin_base",
        "wall_n",
        half_extents=(0.06, 0.005, 0.025),
        local_pos=(0.0, -0.06, 0.025),
        rgba=(0.5, 0.4, 0.3, 1.0),
    )
    add_box_geom(
        stage,
        "/bin/bin_base",
        "wall_s",
        half_extents=(0.06, 0.005, 0.025),
        local_pos=(0.0, 0.06, 0.025),
        rgba=(0.5, 0.4, 0.3, 1.0),
    )
    add_box_geom(
        stage,
        "/bin/bin_base",
        "wall_e",
        half_extents=(0.005, 0.06, 0.025),
        local_pos=(0.06, 0.0, 0.025),
        rgba=(0.5, 0.4, 0.3, 1.0),
    )
    add_box_geom(
        stage,
        "/bin/bin_base",
        "wall_w",
        half_extents=(0.005, 0.06, 0.025),
        local_pos=(-0.06, 0.0, 0.025),
        rgba=(0.5, 0.4, 0.3, 1.0),
    )
    add_handle_marker(stage, "/bin/bin_center", world_pos=(base_pos[0], base_pos[1], base_pos[2] + 0.025))
    add_fixed_joint(
        stage,
        "/bin/center_to_base",
        body0_path="/bin/bin_base",
        body1_path="/bin/bin_center",
        local_pos0=(0.0, 0.0, 0.025),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_fixed_joint_to_world(stage, "/bin/base_to_world", body1_path="/bin/bin_base", world_pos=base_pos)
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Assembly peg-stand (kinematic; ring is a separate RigidObjectCfg)
# ============================================================================


def build_assembly_peg() -> None:
    """A vertical peg (3 cm tall, ~1 cm radius) on a base. Used by:

    * assembly: the agent picks up the ring and places it onto the peg.
    * disassemble: the ring starts on the peg and is lifted off.

    For peg-unplug-side use ``build_peg_unplug()`` — that one is a horizontal
    peg that gets pulled out of a wall via a prismatic joint."""
    out = OUT_ROOT / "assembly_peg" / "usd" / "mw_assembly_peg.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/assembly_peg", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.75, 0.025)
    add_rigid_body_anchor(stage, "/assembly_peg/peg_base", translate=base_pos, mass=5.0)
    add_box_geom(
        stage,
        "/assembly_peg/peg_base",
        "stand",
        half_extents=(0.06, 0.06, 0.025),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.4, 0.4, 0.4, 1.0),
    )
    add_cylinder_geom(
        stage,
        "/assembly_peg/peg_base",
        "peg",
        radius=0.012,
        height=0.06,
        axis="Z",
        local_pos=(0.0, 0.0, 0.055),
        rgba=(0.85, 0.6, 0.3, 1.0),
    )
    add_handle_marker(stage, "/assembly_peg/peg_tip", world_pos=(base_pos[0], base_pos[1], base_pos[2] + 0.085))
    add_fixed_joint(
        stage,
        "/assembly_peg/tip_to_base",
        body0_path="/assembly_peg/peg_base",
        body1_path="/assembly_peg/peg_tip",
        local_pos0=(0.0, 0.0, 0.085),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_fixed_joint_to_world(
        stage, "/assembly_peg/base_to_world", body1_path="/assembly_peg/peg_base", world_pos=base_pos
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Box-with-Lid (revolute lid; box-close)
# ============================================================================


def build_box_with_lid() -> None:
    """Box base with a hinged lid that rotates around its rear edge.
    Box-close success = lid rotated to the closed position (angle = 0)."""
    out = OUT_ROOT / "box_with_lid" / "usd" / "mw_box_with_lid.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/box_with_lid", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.75, 0.04)
    add_rigid_body_anchor(stage, "/box_with_lid/box_base", translate=base_pos, mass=3.0)
    # Box base: bottom + 4 walls (no top — that's the lid).
    add_box_geom(
        stage,
        "/box_with_lid/box_base",
        "bottom",
        half_extents=(0.07, 0.07, 0.005),
        local_pos=(0.0, 0.0, -0.035),
        rgba=(0.6, 0.4, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/box_with_lid/box_base",
        "wall_n",
        half_extents=(0.07, 0.005, 0.04),
        local_pos=(0.0, -0.07, 0.0),
        rgba=(0.6, 0.4, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/box_with_lid/box_base",
        "wall_s",
        half_extents=(0.07, 0.005, 0.04),
        local_pos=(0.0, 0.07, 0.0),
        rgba=(0.6, 0.4, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/box_with_lid/box_base",
        "wall_e",
        half_extents=(0.005, 0.07, 0.04),
        local_pos=(0.07, 0.0, 0.0),
        rgba=(0.6, 0.4, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/box_with_lid/box_base",
        "wall_w",
        half_extents=(0.005, 0.07, 0.04),
        local_pos=(-0.07, 0.0, 0.0),
        rgba=(0.6, 0.4, 0.2, 1.0),
    )
    add_fixed_joint_to_world(
        stage, "/box_with_lid/base_to_world", body1_path="/box_with_lid/box_base", world_pos=base_pos
    )

    # Lid hinged on the rear (north) edge.
    lid_pos = (0.0, 0.75, 0.085)
    add_rigid_body_anchor(stage, "/box_with_lid/lid_link", translate=lid_pos, mass=0.20)
    add_box_geom(
        stage,
        "/box_with_lid/lid_link",
        "panel",
        half_extents=(0.07, 0.07, 0.005),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.7, 0.5, 0.25, 1.0),
    )
    add_handle_marker(stage, "/box_with_lid/lid_marker", world_pos=lid_pos)
    add_fixed_joint(
        stage,
        "/box_with_lid/marker_to_lid",
        body0_path="/box_with_lid/lid_link",
        body1_path="/box_with_lid/lid_marker",
        local_pos0=(0.0, 0.0, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    # Hinge axis = X (rotates around east-west axis at the rear edge).
    add_revolute_joint(
        stage,
        "/box_with_lid/lid_hinge",
        body0_path="/box_with_lid/box_base",
        body1_path="/box_with_lid/lid_link",
        local_pos0=(0.0, -0.07, 0.045),
        local_pos1=(0.0, -0.07, 0.0),
        axis="X",
        lower_deg=0.0,
        upper_deg=120.0,
        damping=2.0,
        stiffness=0.0,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Shelf (kinematic elevated platform; shelf-place)
# ============================================================================


def build_shelf() -> None:
    """Elevated platform. Shelf-place success = cube placed on top of shelf."""
    out = OUT_ROOT / "shelf" / "usd" / "mw_shelf.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/shelf", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.85, 0.10)
    add_rigid_body_anchor(stage, "/shelf/shelf_body", translate=base_pos, mass=5.0)
    # Two vertical posts + a top platform.
    add_box_geom(
        stage,
        "/shelf/shelf_body",
        "post_l",
        half_extents=(0.005, 0.04, 0.10),
        local_pos=(-0.10, 0.0, 0.0),
        rgba=(0.55, 0.35, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/shelf/shelf_body",
        "post_r",
        half_extents=(0.005, 0.04, 0.10),
        local_pos=(+0.10, 0.0, 0.0),
        rgba=(0.55, 0.35, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/shelf/shelf_body",
        "top",
        half_extents=(0.12, 0.06, 0.005),
        local_pos=(0.0, 0.0, 0.10),
        rgba=(0.65, 0.45, 0.25, 1.0),
    )
    add_handle_marker(stage, "/shelf/shelf_top", world_pos=(base_pos[0], base_pos[1], base_pos[2] + 0.105))
    add_fixed_joint(
        stage,
        "/shelf/top_to_body",
        body0_path="/shelf/shelf_body",
        body1_path="/shelf/shelf_top",
        local_pos0=(0.0, 0.0, 0.105),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_fixed_joint_to_world(stage, "/shelf/body_to_world", body1_path="/shelf/shelf_body", world_pos=base_pos)
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Soccer goal (kinematic; soccer)
# ============================================================================


def build_soccer_goal() -> None:
    """Open-front goal. Soccer success = ball inside goal volume."""
    out = OUT_ROOT / "soccer_goal" / "usd" / "mw_soccer_goal.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/soccer_goal", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.85, 0.06)
    add_rigid_body_anchor(stage, "/soccer_goal/goal_body", translate=base_pos, mass=2.0)
    # Two side posts + crossbar + back. Front (toward robot, -y) is open.
    add_box_geom(
        stage,
        "/soccer_goal/goal_body",
        "post_l",
        half_extents=(0.005, 0.04, 0.06),
        local_pos=(-0.10, 0.0, 0.0),
        rgba=(0.95, 0.95, 0.95, 1.0),
    )
    add_box_geom(
        stage,
        "/soccer_goal/goal_body",
        "post_r",
        half_extents=(0.005, 0.04, 0.06),
        local_pos=(0.10, 0.0, 0.0),
        rgba=(0.95, 0.95, 0.95, 1.0),
    )
    add_box_geom(
        stage,
        "/soccer_goal/goal_body",
        "cross",
        half_extents=(0.10, 0.04, 0.005),
        local_pos=(0.0, 0.0, 0.06),
        rgba=(0.95, 0.95, 0.95, 1.0),
    )
    add_box_geom(
        stage,
        "/soccer_goal/goal_body",
        "back",
        half_extents=(0.10, 0.005, 0.06),
        local_pos=(0.0, 0.04, 0.0),
        rgba=(0.95, 0.95, 0.95, 1.0),
    )
    add_handle_marker(stage, "/soccer_goal/goal_center", world_pos=(base_pos[0], base_pos[1] + 0.02, base_pos[2]))
    add_fixed_joint(
        stage,
        "/soccer_goal/center_to_body",
        body0_path="/soccer_goal/goal_body",
        body1_path="/soccer_goal/goal_center",
        local_pos0=(0.0, 0.02, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_fixed_joint_to_world(
        stage, "/soccer_goal/body_to_world", body1_path="/soccer_goal/goal_body", world_pos=base_pos
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Nail block (block + nail on prismatic Z; hammer task)
# ============================================================================


def build_nail_block() -> None:
    """Block with a vertical nail on a prismatic joint. Hammer task = strike
    the nail to drive it down into the block."""
    out = OUT_ROOT / "nail_block" / "usd" / "mw_nail_block.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/nail_block", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.80, 0.04)
    add_rigid_body_anchor(stage, "/nail_block/block_base", translate=base_pos, mass=5.0)
    add_box_geom(
        stage,
        "/nail_block/block_base",
        "block",
        half_extents=(0.05, 0.05, 0.04),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.5, 0.3, 0.2, 1.0),
    )
    add_fixed_joint_to_world(
        stage, "/nail_block/base_to_world", body1_path="/nail_block/block_base", world_pos=base_pos
    )

    # Nail: thin vertical cylinder, slides down (-Z) when struck.
    nail_pos = (0.0, 0.80, 0.10)
    add_rigid_body_anchor(stage, "/nail_block/nail_link", translate=nail_pos, mass=0.05)
    add_cylinder_geom(
        stage,
        "/nail_block/nail_link",
        "shaft",
        radius=0.005,
        height=0.04,
        axis="Z",
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.6, 0.6, 0.65, 1.0),
    )
    add_box_geom(
        stage,
        "/nail_block/nail_link",
        "head",
        half_extents=(0.012, 0.012, 0.003),
        local_pos=(0.0, 0.0, 0.02),
        rgba=(0.6, 0.6, 0.65, 1.0),
    )
    add_handle_marker(stage, "/nail_block/nail_head", world_pos=(nail_pos[0], nail_pos[1], nail_pos[2] + 0.02))
    add_fixed_joint(
        stage,
        "/nail_block/head_to_link",
        body0_path="/nail_block/nail_link",
        body1_path="/nail_block/nail_head",
        local_pos0=(0.0, 0.0, 0.02),
        local_pos1=(0.0, 0.0, 0.0),
    )
    # Prismatic Z, range [-0.06, 0]. 0 = sticking up, -0.06 = driven flush.
    add_prismatic_joint(
        stage,
        "/nail_block/nail_drive",
        body0_path="/nail_block/block_base",
        body1_path="/nail_block/nail_link",
        local_pos0=(0.0, 0.0, 0.06),
        local_pos1=(0.0, 0.0, 0.0),
        axis="Z",
        lower=-0.06,
        upper=0.0,
        damping=2.0,
        stiffness=0.0,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Hole block (kinematic block with cube-shaped pocket; pick-out-of-hole, hand-insert)
# ============================================================================


def build_hole_block() -> None:
    """A solid block with a cube-shaped pocket on top. Used by:

    * pick-out-of-hole: cube starts in the pocket, agent must lift it out.
    * hand-insert: gripper must insert into the pocket itself."""
    out = OUT_ROOT / "hole_block" / "usd" / "mw_hole_block.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/hole_block", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.75, 0.04)
    add_rigid_body_anchor(stage, "/hole_block/block_base", translate=base_pos, mass=5.0)
    # Build wall around a 5×5 cm pocket centered at top. Pocket depth ~6 cm.
    add_box_geom(
        stage,
        "/hole_block/block_base",
        "bottom",
        half_extents=(0.06, 0.06, 0.005),
        local_pos=(0.0, 0.0, -0.035),
        rgba=(0.5, 0.5, 0.55, 1.0),
    )
    add_box_geom(
        stage,
        "/hole_block/block_base",
        "wall_n",
        half_extents=(0.06, 0.005, 0.04),
        local_pos=(0.0, -0.06, 0.0),
        rgba=(0.5, 0.5, 0.55, 1.0),
    )
    add_box_geom(
        stage,
        "/hole_block/block_base",
        "wall_s",
        half_extents=(0.06, 0.005, 0.04),
        local_pos=(0.0, 0.06, 0.0),
        rgba=(0.5, 0.5, 0.55, 1.0),
    )
    add_box_geom(
        stage,
        "/hole_block/block_base",
        "wall_e",
        half_extents=(0.005, 0.06, 0.04),
        local_pos=(0.06, 0.0, 0.0),
        rgba=(0.5, 0.5, 0.55, 1.0),
    )
    add_box_geom(
        stage,
        "/hole_block/block_base",
        "wall_w",
        half_extents=(0.005, 0.06, 0.04),
        local_pos=(-0.06, 0.0, 0.0),
        rgba=(0.5, 0.5, 0.55, 1.0),
    )
    add_handle_marker(stage, "/hole_block/hole_marker", world_pos=(base_pos[0], base_pos[1], base_pos[2] - 0.025))
    add_fixed_joint(
        stage,
        "/hole_block/marker_to_base",
        body0_path="/hole_block/block_base",
        body1_path="/hole_block/hole_marker",
        local_pos0=(0.0, 0.0, -0.025),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_fixed_joint_to_world(
        stage, "/hole_block/base_to_world", body1_path="/hole_block/block_base", world_pos=base_pos
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Peg-Unplug (block with horizontal peg pre-inserted on prismatic X)
# ============================================================================


def build_peg_unplug() -> None:
    """Wall-mounted peg that the agent pulls out along world X. Distinct
    from peg-insert-side: there the peg is free and gets inserted; here the
    peg is articulated and gets pulled out (or pushed back in)."""
    out = OUT_ROOT / "peg_unplug" / "usd" / "mw_peg_unplug.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/peg_unplug", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.85, 0.10)
    add_rigid_body_anchor(stage, "/peg_unplug/wall_base", translate=base_pos, mass=5.0)
    add_box_geom(
        stage,
        "/peg_unplug/wall_base",
        "wall",
        half_extents=(0.04, 0.06, 0.10),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.45, 0.30, 0.20, 1.0),
    )
    add_fixed_joint_to_world(stage, "/peg_unplug/wall_to_world", body1_path="/peg_unplug/wall_base", world_pos=base_pos)

    # Peg sticks out toward -X (toward robot), slides along X.
    peg_pos = (-0.05, 0.85, 0.10)
    add_rigid_body_anchor(stage, "/peg_unplug/peg_link", translate=peg_pos, mass=0.05)
    add_cylinder_geom(
        stage,
        "/peg_unplug/peg_link",
        "peg",
        radius=0.012,
        height=0.06,
        axis="X",
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.85, 0.6, 0.3, 1.0),
    )
    add_handle_marker(stage, "/peg_unplug/peg_tip", world_pos=(peg_pos[0] - 0.03, peg_pos[1], peg_pos[2]))
    add_fixed_joint(
        stage,
        "/peg_unplug/tip_to_link",
        body0_path="/peg_unplug/peg_link",
        body1_path="/peg_unplug/peg_tip",
        local_pos0=(-0.03, 0.0, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    # Range [-0.10, 0]: 0 = peg fully inserted, -0.10 = peg fully pulled out.
    add_prismatic_joint(
        stage,
        "/peg_unplug/peg_slide",
        body0_path="/peg_unplug/wall_base",
        body1_path="/peg_unplug/peg_link",
        local_pos0=(-0.05, 0.0, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
        axis="X",
        lower=-0.10,
        upper=0.0,
        damping=1.0,
        stiffness=0.0,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Plate-side (slides along world Y instead of X — for plate-slide-side)
# ============================================================================


def build_plate_side() -> None:
    """Same shape as ``build_plate`` but with the groove + prismatic joint
    rotated 90°: rails run along world Y, plate slides along Y.

    Used by plate-slide-side and plate-slide-back-side. Authoring a separate
    USD (rather than rotating the plate via ``init_state.rot``) avoids the
    ambiguity of how PhysX interprets a rotated articulation root with an
    internal fixed-joint-to-world.
    """
    out = OUT_ROOT / "plate_side" / "usd" / "mw_plate_side.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/plate", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.75, 0.02)
    add_rigid_body_anchor(stage, "/plate/groove_base", translate=base_pos, mass=2.0)
    # Groove rails along world Y (rotated 90° from build_plate's X-aligned groove).
    add_box_geom(
        stage,
        "/plate/groove_base",
        "bottom",
        half_extents=(0.06, 0.20, 0.005),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.45, 0.5, 0.55, 1.0),
    )
    add_box_geom(
        stage,
        "/plate/groove_base",
        "rail_w",
        half_extents=(0.005, 0.20, 0.015),
        local_pos=(-0.06, 0.0, 0.015),
        rgba=(0.45, 0.5, 0.55, 1.0),
    )
    add_box_geom(
        stage,
        "/plate/groove_base",
        "rail_e",
        half_extents=(0.005, 0.20, 0.015),
        local_pos=(+0.06, 0.0, 0.015),
        rgba=(0.45, 0.5, 0.55, 1.0),
    )
    add_fixed_joint_to_world(stage, "/plate/base_to_world", body1_path="/plate/groove_base", world_pos=base_pos)

    # Plate sits at -y end of groove at joint=0.
    plate_pos = (0.0, 0.65, 0.025)
    add_rigid_body_anchor(stage, "/plate/plate_link", translate=plate_pos, mass=0.10)
    add_box_geom(
        stage,
        "/plate/plate_link",
        "puck",
        half_extents=(0.04, 0.04, 0.01),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.85, 0.55, 0.20, 1.0),
    )
    add_handle_marker(stage, "/plate/plate_marker", world_pos=plate_pos)
    add_fixed_joint(
        stage,
        "/plate/marker_to_link",
        body0_path="/plate/plate_link",
        body1_path="/plate/plate_marker",
        local_pos0=(0.0, 0.0, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_prismatic_joint(
        stage,
        "/plate/plate_slide",
        body0_path="/plate/groove_base",
        body1_path="/plate/plate_link",
        local_pos0=(0.0, -0.10, 0.005),
        local_pos1=(0.0, 0.0, 0.0),
        axis="Y",
        lower=-0.20,
        upper=0.20,
        damping=0.5,
        stiffness=0.0,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


# ============================================================================
# Button-front (front-facing button on a wall mount, prismatic Y)
# ============================================================================


def build_button_front() -> None:
    """Front-facing button: cap protrudes toward the agent (-Y) and is pressed
    by pushing it +Y (into the wall mount). Used by button-press and
    button-press-wall.

    Joint convention: range ``[0, 0.06]`` where ``joint=0`` = extended (cap
    protruded) and ``joint=+0.06`` = pressed (cap retracted into mount).
    Different from the topdown button's ``[-0.06, 0]`` because the prismatic
    axis points away from the agent — see the env spec for goal joint values.
    """
    out = OUT_ROOT / "button_front" / "usd" / "mw_button_front.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/button", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    # Wall-mounted base behind the button (at world y=0.85).
    base_pos = (0.0, 0.85, 0.10)
    add_rigid_body_anchor(stage, "/button/button_box", translate=base_pos, mass=1.0)
    add_box_geom(
        stage,
        "/button/button_box",
        "stand",
        half_extents=(0.06, 0.05, 0.10),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.4, 0.4, 0.4, 1.0),
    )
    add_fixed_joint_to_world(stage, "/button/box_to_world", body1_path="/button/button_box", world_pos=base_pos)

    # Cap protrudes toward agent (-Y) at joint=0.
    button_pos = (0.0, 0.79, 0.10)
    add_rigid_body_anchor(stage, "/button/button_link", translate=button_pos, mass=0.05)
    add_box_geom(
        stage,
        "/button/button_link",
        "cap",
        half_extents=(0.03, 0.015, 0.03),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.9, 0.1, 0.1, 1.0),
    )
    add_handle_marker(stage, "/button/button_top", world_pos=button_pos)
    add_fixed_joint(
        stage,
        "/button/top_to_link",
        body0_path="/button/button_link",
        body1_path="/button/button_top",
        local_pos0=(0.0, 0.0, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    # Prismatic axis Y, range [0, 0.06]. joint=0 → cap at world (0, 0.79, 0.10).
    # joint=+0.06 → cap at (0, 0.85, 0.10) (retracted into mount).
    add_prismatic_joint(
        stage,
        "/button/btnbox_joint",
        body0_path="/button/button_box",
        body1_path="/button/button_link",
        local_pos0=(0.0, -0.06, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
        axis="Y",
        lower=0.0,
        upper=0.06,
        damping=1.0,
        stiffness=0.5,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets+] wrote {out}")


def main() -> None:
    build_wall()
    build_handle_press()
    build_handle_press_side()
    build_plate()
    build_basket()
    build_bin()
    build_assembly_peg()
    build_box_with_lid()
    build_shelf()
    build_soccer_goal()
    build_nail_block()
    build_hole_block()
    build_peg_unplug()
    build_plate_side()
    build_button_front()


if __name__ == "__main__":
    main()
    simulation_app.close()
