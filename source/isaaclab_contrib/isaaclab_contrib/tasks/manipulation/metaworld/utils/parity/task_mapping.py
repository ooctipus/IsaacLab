# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mapping between Meta-World v3 task names and IsaacLab gym IDs / spec keys.

The MW dump uses ``<task>-v3`` strings (e.g. ``drawer-open-v3``); the
IsaacLab side uses ``Isaac-Metaworld-<Task>-Sawyer-v0`` and the
:data:`TASK_SPECS` keys use ``<task>`` (e.g. ``drawer_open``).
"""

from __future__ import annotations

# MW name → (IsaacLab gym ID, TASK_SPECS key or None for MT3 cube tasks).
# When TASK_SPECS key is None the task is MT3 (cube IS the manipulandum):
# placement parity is checked against the cube/goal pose, not a welded
# articulated marker.
MW_TO_ISAAC: dict[str, tuple[str, str | None]] = {
    # MT3 (cube manipulandum)
    "reach-v3": ("Isaac-Metaworld-Reach-Sawyer-v0", None),
    "push-v3": ("Isaac-Metaworld-Push-Sawyer-v0", None),
    "pick-place-v3": ("Isaac-Metaworld-Pick-Place-Sawyer-v0", None),
    # 15 articulated real-asset
    "drawer-open-v3": ("Isaac-Metaworld-Drawer-Open-Sawyer-v0", "drawer_open"),
    "drawer-close-v3": ("Isaac-Metaworld-Drawer-Close-Sawyer-v0", "drawer_close"),
    "button-press-topdown-v3": ("Isaac-Metaworld-Button-Press-Topdown-Sawyer-v0", "button_press_topdown"),
    "coffee-button-v3": ("Isaac-Metaworld-Coffee-Button-Sawyer-v0", "coffee_button"),
    "window-open-v3": ("Isaac-Metaworld-Window-Open-Sawyer-v0", "window_open"),
    "window-close-v3": ("Isaac-Metaworld-Window-Close-Sawyer-v0", "window_close"),
    "faucet-open-v3": ("Isaac-Metaworld-Faucet-Open-Sawyer-v0", "faucet_open"),
    "faucet-close-v3": ("Isaac-Metaworld-Faucet-Close-Sawyer-v0", "faucet_close"),
    "dial-turn-v3": ("Isaac-Metaworld-Dial-Turn-Sawyer-v0", "dial_turn"),
    "lever-pull-v3": ("Isaac-Metaworld-Lever-Pull-Sawyer-v0", "lever_pull"),
    "door-open-v3": ("Isaac-Metaworld-Door-Open-Sawyer-v0", "door_open"),
    "door-close-v3": ("Isaac-Metaworld-Door-Close-Sawyer-v0", "door_close"),
    "door-lock-v3": ("Isaac-Metaworld-Door-Lock-Sawyer-v0", "door_lock"),
    "door-unlock-v3": ("Isaac-Metaworld-Door-Unlock-Sawyer-v0", "door_unlock"),
    "peg-insert-side-v3": ("Isaac-Metaworld-Peg-Insert-Side-Sawyer-v0", "peg_insert_side"),
    # MT50 expansion — cube + obstacle/destination
    "push-back-v3": ("Isaac-Metaworld-Push-Back-Sawyer-v0", "push_back"),
    "push-wall-v3": ("Isaac-Metaworld-Push-Wall-Sawyer-v0", "push_wall"),
    "reach-wall-v3": ("Isaac-Metaworld-Reach-Wall-Sawyer-v0", "reach_wall"),
    "pick-place-wall-v3": ("Isaac-Metaworld-Pick-Place-Wall-Sawyer-v0", "pick_place_wall"),
    "basketball-v3": ("Isaac-Metaworld-Basketball-Sawyer-v0", "basketball"),
    "shelf-place-v3": ("Isaac-Metaworld-Shelf-Place-Sawyer-v0", "shelf_place"),
    "soccer-v3": ("Isaac-Metaworld-Soccer-Sawyer-v0", "soccer"),
    "sweep-v3": ("Isaac-Metaworld-Sweep-Sawyer-v0", "sweep"),
    "sweep-into-v3": ("Isaac-Metaworld-Sweep-Into-Sawyer-v0", "sweep_into"),
    "plate-slide-v3": ("Isaac-Metaworld-Plate-Slide-Sawyer-v0", "plate_slide"),
    "plate-slide-back-v3": ("Isaac-Metaworld-Plate-Slide-Back-Sawyer-v0", "plate_slide_back"),
    "handle-press-v3": ("Isaac-Metaworld-Handle-Press-Sawyer-v0", "handle_press"),
    "handle-pull-v3": ("Isaac-Metaworld-Handle-Pull-Sawyer-v0", "handle_pull"),
    "handle-press-side-v3": ("Isaac-Metaworld-Handle-Press-Side-Sawyer-v0", "handle_press_side"),
    "handle-pull-side-v3": ("Isaac-Metaworld-Handle-Pull-Side-Sawyer-v0", "handle_pull_side"),
    "peg-unplug-side-v3": ("Isaac-Metaworld-Peg-Unplug-Side-Sawyer-v0", "peg_unplug_side"),
    "box-close-v3": ("Isaac-Metaworld-Box-Close-Sawyer-v0", "box_close"),
    "hand-insert-v3": ("Isaac-Metaworld-Hand-Insert-Sawyer-v0", "hand_insert"),
    "pick-out-of-hole-v3": ("Isaac-Metaworld-Pick-Out-Of-Hole-Sawyer-v0", "pick_out_of_hole"),
    "button-press-topdown-wall-v3": (
        "Isaac-Metaworld-Button-Press-Topdown-Wall-Sawyer-v0",
        "button_press_topdown_wall",
    ),
    "bin-picking-v3": ("Isaac-Metaworld-Bin-Picking-Sawyer-v0", "bin_picking"),
    "coffee-push-v3": ("Isaac-Metaworld-Coffee-Push-Sawyer-v0", "coffee_push"),
    "coffee-pull-v3": ("Isaac-Metaworld-Coffee-Pull-Sawyer-v0", "coffee_pull"),
    "stick-push-v3": ("Isaac-Metaworld-Stick-Push-Sawyer-v0", "stick_push"),
    "stick-pull-v3": ("Isaac-Metaworld-Stick-Pull-Sawyer-v0", "stick_pull"),
    "assembly-v3": ("Isaac-Metaworld-Assembly-Sawyer-v0", "assembly"),
    "disassemble-v3": ("Isaac-Metaworld-Disassemble-Sawyer-v0", "disassemble"),
    "hammer-v3": ("Isaac-Metaworld-Hammer-Sawyer-v0", "hammer"),
    "plate-slide-side-v3": ("Isaac-Metaworld-Plate-Slide-Side-Sawyer-v0", "plate_slide_side"),
    "plate-slide-back-side-v3": (
        "Isaac-Metaworld-Plate-Slide-Back-Side-Sawyer-v0",
        "plate_slide_back_side",
    ),
    "button-press-v3": ("Isaac-Metaworld-Button-Press-Sawyer-v0", "button_press"),
    "button-press-wall-v3": ("Isaac-Metaworld-Button-Press-Wall-Sawyer-v0", "button_press_wall"),
}


def isaac_id(mw_name: str) -> str:
    return MW_TO_ISAAC[mw_name][0]


def spec_key(mw_name: str) -> str | None:
    return MW_TO_ISAAC[mw_name][1]


__all__ = ["MW_TO_ISAAC", "isaac_id", "spec_key"]
