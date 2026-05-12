# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Author all hand-built MW-style task assets in one pass.

Outputs:

* ``door/usd/mw_door.usda``  — revolute door panel + handle.
* ``button/usd/mw_button.usda`` — cylinder + prismatic (downward push).
* ``window/usd/mw_window.usda`` — sliding window with handle.
* ``faucet/usd/mw_faucet.usda`` — handle on revolute joint.
* ``peg/usd/mw_peg.usda`` — perforated wall + free peg cylinder.

Each USD follows the same flat-sibling pattern as ``mw_drawer.usda``:
kinematic base body welded to world, moving body with the actuated
joint, a zero-extent handle/marker rigid body for FrameTransformer.

Geometry is simplified vs MW's MJCF (no mesh imports) but matches
the relevant surfaces, joint axes/ranges, and handle/marker positions
needed for reward terms and scripted-policy verification.
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
    print(f"[mw_assets] wrote {out}")


# ============================================================================
# Door (revolute door panel + handle)
# ============================================================================


def build_door() -> None:
    """MW door reference: ``sawyer_door.xml`` — door body at world (0, 0.85, 0.15)
    rotated; door_link rotates around vertical axis; handle protrudes."""
    out = OUT_ROOT / "door" / "usd" / "mw_door.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/door", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    # base frame (door jamb) at world (0, 0.95, 0.15)
    base_pos = (0.0, 0.95, 0.15)
    add_rigid_body_anchor(stage, "/door/door_frame", translate=base_pos, mass=1.0)
    # frame walls — left + right + top
    add_box_geom(
        stage,
        "/door/door_frame",
        "left",
        half_extents=(0.005, 0.05, 0.15),
        local_pos=(-0.20, 0.0, 0.0),
        rgba=(0.5, 0.3, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/door/door_frame",
        "right",
        half_extents=(0.005, 0.05, 0.15),
        local_pos=(+0.20, 0.0, 0.0),
        rgba=(0.5, 0.3, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/door/door_frame",
        "top",
        half_extents=(0.20, 0.05, 0.005),
        local_pos=(0.0, 0.0, +0.15),
        rgba=(0.5, 0.3, 0.2, 1.0),
    )
    add_fixed_joint_to_world(stage, "/door/frame_to_world", body1_path="/door/door_frame", world_pos=base_pos)

    # door_link — panel that rotates around left-frame's vertical axis
    door_link_pos = (0.0, 0.95, 0.15)
    add_rigid_body_anchor(stage, "/door/door_link", translate=door_link_pos, mass=0.5)
    add_box_geom(
        stage,
        "/door/door_link",
        "panel",
        half_extents=(0.18, 0.01, 0.13),
        local_pos=(0.18, 0.0, 0.0),
        rgba=(0.6, 0.4, 0.25, 1.0),
    )
    # handle on the door (right side, since hinge is on the left)
    add_box_geom(
        stage,
        "/door/door_link",
        "handle_post",
        half_extents=(0.005, 0.04, 0.005),
        local_pos=(0.32, -0.04, 0.0),
        rgba=(0.9, 0.9, 0.9, 1.0),
    )
    add_box_geom(
        stage,
        "/door/door_link",
        "handle_bar",
        half_extents=(0.04, 0.005, 0.005),
        local_pos=(0.32, -0.08, 0.0),
        rgba=(0.9, 0.9, 0.9, 1.0),
    )
    add_handle_marker(stage, "/door/door_handle", world_pos=(0.32, 0.95 - 0.08, 0.15))
    add_fixed_joint(
        stage,
        "/door/handle_to_link",
        body0_path="/door/door_link",
        body1_path="/door/door_handle",
        local_pos0=(0.32, -0.08, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    # revolute joint at the LEFT side of the door (x=0, y=0 in door_link frame)
    add_revolute_joint(
        stage,
        "/door/door_hinge",
        body0_path="/door/door_frame",
        body1_path="/door/door_link",
        local_pos0=(0.0, 0.0, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
        axis="Z",
        lower_deg=-90.0,
        upper_deg=0.0,
        damping=2.0,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets] wrote {out}")


# ============================================================================
# Button (vertical cylinder on prismatic joint, downward push)
# ============================================================================


def build_button() -> None:
    out = OUT_ROOT / "button" / "usd" / "mw_button.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/button", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.85, 0.05)
    add_rigid_body_anchor(stage, "/button/button_box", translate=base_pos, mass=1.0)
    # button box stand
    add_box_geom(
        stage,
        "/button/button_box",
        "base",
        half_extents=(0.06, 0.06, 0.05),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.4, 0.4, 0.4, 1.0),
    )
    add_fixed_joint_to_world(stage, "/button/box_to_world", body1_path="/button/button_box", world_pos=base_pos)

    # button cap — slides downward (-Z) when pressed
    button_pos = (0.0, 0.85, 0.115)
    add_rigid_body_anchor(stage, "/button/button_link", translate=button_pos, mass=0.05)
    add_box_geom(
        stage,
        "/button/button_link",
        "cap",
        half_extents=(0.03, 0.03, 0.015),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.9, 0.1, 0.1, 1.0),
    )
    add_handle_marker(stage, "/button/button_top", world_pos=button_pos)
    add_fixed_joint(
        stage,
        "/button/top_to_link",
        body0_path="/button/button_link",
        body1_path="/button/button_top",
        local_pos0=(0.0, 0.0, 0.015),
        local_pos1=(0.0, 0.0, 0.0),
    )
    # prismatic in -Z, range [-0.06, 0]: 0 = fully extended, -0.06 = fully pressed
    add_prismatic_joint(
        stage,
        "/button/btnbox_joint",
        body0_path="/button/button_box",
        body1_path="/button/button_link",
        local_pos0=(0.0, 0.0, 0.065),
        local_pos1=(0.0, 0.0, 0.0),
        axis="Z",
        lower=-0.06,
        upper=0.0,
        damping=1.0,
        stiffness=0.5,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets] wrote {out}")


# ============================================================================
# Window (horizontal slider window with handle)
# ============================================================================


def build_window() -> None:
    out = OUT_ROOT / "window" / "usd" / "mw_window.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/window", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.785, 0.10)
    add_rigid_body_anchor(stage, "/window/window_frame", translate=base_pos, mass=1.0)
    # frame: bottom and top rails
    add_box_geom(
        stage,
        "/window/window_frame",
        "bottom_rail",
        half_extents=(0.20, 0.03, 0.01),
        local_pos=(0.0, 0.0, -0.06),
        rgba=(0.3, 0.5, 0.3, 1.0),
    )
    add_box_geom(
        stage,
        "/window/window_frame",
        "top_rail",
        half_extents=(0.20, 0.03, 0.01),
        local_pos=(0.0, 0.0, +0.06),
        rgba=(0.3, 0.5, 0.3, 1.0),
    )
    add_box_geom(
        stage,
        "/window/window_frame",
        "left_post",
        half_extents=(0.01, 0.03, 0.05),
        local_pos=(-0.20, 0.0, 0.0),
        rgba=(0.3, 0.5, 0.3, 1.0),
    )
    add_box_geom(
        stage,
        "/window/window_frame",
        "right_post",
        half_extents=(0.01, 0.03, 0.05),
        local_pos=(+0.20, 0.0, 0.0),
        rgba=(0.3, 0.5, 0.3, 1.0),
    )
    add_fixed_joint_to_world(stage, "/window/frame_to_world", body1_path="/window/window_frame", world_pos=base_pos)

    # window slider (slides along X)
    slider_pos = (-0.10, 0.785, 0.10)  # closed = at -x
    add_rigid_body_anchor(stage, "/window/window_link", translate=slider_pos, mass=0.05)
    add_box_geom(
        stage,
        "/window/window_link",
        "pane",
        half_extents=(0.08, 0.005, 0.04),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.7, 0.85, 0.95, 0.5),
    )
    # handle on the pane
    add_box_geom(
        stage,
        "/window/window_link",
        "handle",
        half_extents=(0.012, 0.015, 0.012),
        local_pos=(0.06, -0.02, 0.0),
        rgba=(0.2, 0.2, 0.2, 1.0),
    )
    add_handle_marker(
        stage, "/window/window_handle", world_pos=(slider_pos[0] + 0.06, slider_pos[1] - 0.02, slider_pos[2])
    )
    add_fixed_joint(
        stage,
        "/window/handle_to_link",
        body0_path="/window/window_link",
        body1_path="/window/window_handle",
        local_pos0=(0.06, -0.02, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_prismatic_joint(
        stage,
        "/window/window_slide",
        body0_path="/window/window_frame",
        body1_path="/window/window_link",
        local_pos0=(-0.10, 0.0, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
        axis="X",
        lower=0.0,
        upper=0.20,
        damping=1.0,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets] wrote {out}")


# ============================================================================
# Faucet (handle on revolute, axis Z)
# ============================================================================


def build_faucet() -> None:
    out = OUT_ROOT / "faucet" / "usd" / "mw_faucet.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/faucet", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (0.0, 0.85, 0.05)
    add_rigid_body_anchor(stage, "/faucet/faucet_base", translate=base_pos, mass=1.0)
    # column and head (matching MW's faucet stack)
    add_box_geom(
        stage,
        "/faucet/faucet_base",
        "base_disk",
        half_extents=(0.046, 0.046, 0.009),
        local_pos=(0.0, 0.0, 0.009),
        rgba=(0.7, 0.7, 0.75, 1.0),
    )
    add_box_geom(
        stage,
        "/faucet/faucet_base",
        "column",
        half_extents=(0.017, 0.017, 0.044),
        local_pos=(0.0, 0.0, 0.061),
        rgba=(0.7, 0.7, 0.75, 1.0),
    )
    add_fixed_joint_to_world(stage, "/faucet/base_to_world", body1_path="/faucet/faucet_base", world_pos=base_pos)

    # handle (rotates around Z above the column)
    handle_pos = (0.0, 0.85, 0.174)  # above the column at z=0.05+0.124
    add_rigid_body_anchor(stage, "/faucet/faucet_handle", translate=handle_pos, mass=0.1)
    # handle is a horizontal bar pointing in +y
    add_box_geom(
        stage,
        "/faucet/faucet_handle",
        "bar",
        half_extents=(0.02, 0.06, 0.02),
        local_pos=(0.0, -0.06, 0.0),
        rgba=(0.85, 0.2, 0.2, 1.0),
    )
    # handle tip marker
    add_handle_marker(stage, "/faucet/handle_tip", world_pos=(handle_pos[0], handle_pos[1] - 0.12, handle_pos[2]))
    add_fixed_joint(
        stage,
        "/faucet/tip_to_handle",
        body0_path="/faucet/faucet_handle",
        body1_path="/faucet/handle_tip",
        local_pos0=(0.0, -0.12, 0.0),
        local_pos1=(0.0, 0.0, 0.0),
    )
    add_revolute_joint(
        stage,
        "/faucet/knob_Joint_1",
        body0_path="/faucet/faucet_base",
        body1_path="/faucet/faucet_handle",
        local_pos0=(0.0, 0.0, 0.124),
        local_pos1=(0.0, 0.0, 0.0),
        axis="Z",
        lower_deg=-90.0,
        upper_deg=90.0,
        damping=2.0,
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets] wrote {out}")


# ============================================================================
# Peg-Insert-Side (perforated wall with hole + free peg cylinder)
# ============================================================================


def build_peg() -> None:
    """Wall with a hole the agent inserts a peg into. Peg is a separate
    free RigidObjectCfg (spawned by the env), not part of this articulation."""
    out = OUT_ROOT / "peg" / "usd" / "mw_peg_block.usda"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    stage = stage_init(out)

    root = stage.DefinePrim("/peg_block", "Xform")
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)

    base_pos = (-0.35, 0.65, 0.10)
    add_rigid_body_anchor(stage, "/peg_block/block", translate=base_pos, mass=10.0)
    # wall with hole at center: build as 4 slab boxes around a central gap
    # Block dims ~0.18 wide × 0.20 deep × 0.20 tall, hole at z=0.13 (3cm above center)
    # Boxes around the hole:
    add_box_geom(
        stage,
        "/peg_block/block",
        "left",
        half_extents=(0.005, 0.10, 0.10),
        local_pos=(-0.04, 0.0, 0.10),
        rgba=(0.6, 0.3, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/peg_block/block",
        "right",
        half_extents=(0.005, 0.10, 0.10),
        local_pos=(+0.04, 0.0, 0.10),
        rgba=(0.6, 0.3, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/peg_block/block",
        "bottom",
        half_extents=(0.04, 0.10, 0.025),
        local_pos=(0.0, 0.0, 0.0),
        rgba=(0.6, 0.3, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/peg_block/block",
        "top",
        half_extents=(0.04, 0.10, 0.025),
        local_pos=(0.0, 0.0, 0.20),
        rgba=(0.6, 0.3, 0.2, 1.0),
    )
    add_box_geom(
        stage,
        "/peg_block/block",
        "back",
        half_extents=(0.04, 0.005, 0.10),
        local_pos=(0.0, 0.10, 0.10),
        rgba=(0.6, 0.3, 0.2, 1.0),
    )
    add_fixed_joint_to_world(stage, "/peg_block/block_to_world", body1_path="/peg_block/block", world_pos=base_pos)
    # marker at hole position (-0.35, 0.65-0.096, 0.13) per MW
    add_handle_marker(stage, "/peg_block/hole", world_pos=(-0.35, 0.65 - 0.096, 0.13))
    add_fixed_joint(
        stage,
        "/peg_block/hole_to_block",
        body0_path="/peg_block/block",
        body1_path="/peg_block/hole",
        local_pos0=(0.0, -0.096, 0.03),
        local_pos1=(0.0, 0.0, 0.0),
    )
    stage.GetRootLayer().Save()
    print(f"[mw_assets] wrote {out}")


def main() -> None:
    build_door()
    build_button()
    build_window()
    build_faucet()
    build_peg()


if __name__ == "__main__":
    main()
    simulation_app.close()
