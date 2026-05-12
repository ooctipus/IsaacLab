# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-task constants — single source of truth for the 15 real-asset tasks.

Inspired by ``factory_v1/assembly_keypoints.py``: instead of scattering
``obj_init`` / ``goal`` / joint reset / success threshold values across
``env_cfgs.py`` (single-task) and ``multi_task_env_cfg.py`` (multi-task),
all four pieces live here in one :class:`MetaworldTaskSpec` per task.

Both env-cfg layers (single-task and multi-task) read from
:data:`TASK_SPECS` so the values can't drift apart.

MT3 (reach / push / pick-place) is intentionally NOT in this table —
those tasks use *uniform rectangle sampling* rather than fixed-pair
sampling, which is a different shape. They live in
``config/sawyer/{reach,push,pick_place}_env_cfg.py`` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .mdp import (
    BUTTON_TARGET_RADIUS,
    DOOR_TARGET_RADIUS,
    DRAWER_TARGET_RADIUS,
    PEG_TARGET_RADIUS,
    PICK_PLACE_SUCCESS_RADIUS,
    PUSH_TARGET_RADIUS,
    REACH_TARGET_RADIUS,
    WINDOW_TARGET_RADIUS,
)


@dataclass(frozen=True)
class MetaworldTaskSpec:
    """Per-task constants for the Meta-World tasks.

    Used by both single-task and multi-task envs to avoid duplicating the
    same coordinates / joint values / thresholds in two places.
    """

    obj_init: tuple[float, float, float]
    """Object initial position [m] in env-local frame — the **mean** of the
    sampling range when ``obj_range_*`` are set; otherwise the fixed point."""

    goal: tuple[float, float, float]
    """Goal position [m] in env-local frame — mean (or fixed) like ``obj_init``."""

    joint_name: str | None
    """Asset joint name to reset, or ``None`` for assets without joints."""

    joint_reset_value: float
    """Joint position [m or rad] to write at episode start.

    For prismatic joints this is metres; for revolute joints, radians."""

    success_threshold: float
    """L2 distance threshold [m] between manipulandum keypoint and goal,
    below which the success indicator fires."""

    obj_range_low: tuple[float, float, float] | None = None
    """Per-env obj_init sampling lower bound [m]. ``None`` collapses to
    ``obj_init`` (fixed point — preserves the original deterministic
    behaviour). Set to MW's per-task ``obj_init_pos`` range to recover the
    Meta-World per-episode randomization (σ ≈ 0.03–0.06 m)."""

    obj_range_high: tuple[float, float, float] | None = None
    """Per-env obj_init sampling upper bound [m] — see :attr:`obj_range_low`."""

    goal_range_low: tuple[float, float, float] | None = None
    """Per-env goal sampling lower bound [m] — see :attr:`obj_range_low`."""

    goal_range_high: tuple[float, float, float] | None = None
    """Per-env goal sampling upper bound [m] — see :attr:`obj_range_low`."""

    def obj_range(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return ``(low, high)``; falls back to ``(obj_init, obj_init)`` when
        ranges are unset (point sampling)."""
        if self.obj_range_low is None or self.obj_range_high is None:
            return self.obj_init, self.obj_init
        return self.obj_range_low, self.obj_range_high

    def goal_range(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return ``(low, high)``; falls back to ``(goal, goal)`` when ranges
        are unset (point sampling)."""
        if self.goal_range_low is None or self.goal_range_high is None:
            return self.goal, self.goal
        return self.goal_range_low, self.goal_range_high


# Convenience: π/3 and π/6 come up several times for door/faucet ranges.
_PI3 = math.pi / 3  # 1.0472
_PI6 = math.pi / 6  # 0.5236


TASK_SPECS: dict[str, MetaworldTaskSpec] = {
    # ─ Drawer ──────────────────────────────────────────────────────────
    "drawer_open": MetaworldTaskSpec(
        obj_init=(0.0, 0.74, 0.09),
        goal=(0.0, 0.58, 0.09),
        joint_name="goal_slidey",
        joint_reset_value=0.0,
        success_threshold=0.03,
    ),
    "drawer_close": MetaworldTaskSpec(
        obj_init=(0.0, 0.58, 0.09),
        goal=(0.0, 0.74, 0.09),
        joint_name="goal_slidey",
        joint_reset_value=-0.16,
        success_threshold=DRAWER_TARGET_RADIUS + 0.015,
    ),
    # ─ Button ──────────────────────────────────────────────────────────
    "button_press_topdown": MetaworldTaskSpec(
        obj_init=(0.0, 0.85, 0.13),
        goal=(0.0, 0.85, 0.07),
        joint_name="btnbox_joint",
        joint_reset_value=0.0,
        success_threshold=BUTTON_TARGET_RADIUS,
    ),
    "coffee_button": MetaworldTaskSpec(
        obj_init=(0.0, 0.85, 0.13),
        goal=(0.0, 0.85, 0.10),
        joint_name="btnbox_joint",
        joint_reset_value=0.0,
        success_threshold=BUTTON_TARGET_RADIUS,
    ),
    # ─ Window ──────────────────────────────────────────────────────────
    "window_open": MetaworldTaskSpec(
        obj_init=(-0.04, 0.765, 0.10),
        goal=(0.16, 0.765, 0.10),
        joint_name="window_slide",
        joint_reset_value=0.0,
        success_threshold=WINDOW_TARGET_RADIUS,
    ),
    "window_close": MetaworldTaskSpec(
        obj_init=(0.16, 0.765, 0.10),
        goal=(-0.04, 0.765, 0.10),
        joint_name="window_slide",
        joint_reset_value=0.20,
        success_threshold=WINDOW_TARGET_RADIUS,
    ),
    # ─ Faucet ──────────────────────────────────────────────────────────
    "faucet_open": MetaworldTaskSpec(
        obj_init=(0.0, 0.73, 0.174),
        goal=(0.104, 0.79, 0.174),
        joint_name="knob_Joint_1",
        joint_reset_value=0.0,
        success_threshold=0.05,
    ),
    "faucet_close": MetaworldTaskSpec(
        # Per parity dump (faucet-close-v3): MW resets ``knob_Joint_1`` at 0
        # (handle starts centred); the close direction is encoded in the goal,
        # which sits at the negative-x rotated tip. Earlier code reset the
        # joint to π/3 ≈ +60° — that was wrong (handle started already open).
        obj_init=(0.104, 0.79, 0.174),
        goal=(0.0, 0.73, 0.174),
        joint_name="knob_Joint_1",
        joint_reset_value=0.0,
        success_threshold=0.05,
    ),
    "dial_turn": MetaworldTaskSpec(
        obj_init=(0.0, 0.73, 0.174),
        goal=(0.06, 0.746, 0.174),
        joint_name="knob_Joint_1",
        joint_reset_value=0.0,
        success_threshold=0.05,
    ),
    "lever_pull": MetaworldTaskSpec(
        obj_init=(-0.104, 0.79, 0.174),
        goal=(0.0, 0.73, 0.174),
        joint_name="knob_Joint_1",
        joint_reset_value=-_PI3,
        success_threshold=0.05,
    ),
    # ─ Door ────────────────────────────────────────────────────────────
    "door_open": MetaworldTaskSpec(
        obj_init=(0.32, 0.87, 0.15),
        goal=(0.091, 0.633, 0.15),
        joint_name="door_hinge",
        joint_reset_value=0.0,
        success_threshold=DOOR_TARGET_RADIUS,
    ),
    "door_close": MetaworldTaskSpec(
        obj_init=(0.091, 0.633, 0.15),
        goal=(0.32, 0.87, 0.15),
        joint_name="door_hinge",
        joint_reset_value=-_PI3,
        success_threshold=DOOR_TARGET_RADIUS,
    ),
    "door_lock": MetaworldTaskSpec(
        obj_init=(0.32, 0.87, 0.15),
        goal=(0.237, 0.721, 0.15),
        joint_name="door_hinge",
        joint_reset_value=0.0,
        success_threshold=0.05,
    ),
    "door_unlock": MetaworldTaskSpec(
        obj_init=(0.237, 0.721, 0.15),
        goal=(0.32, 0.87, 0.15),
        joint_name="door_hinge",
        joint_reset_value=-_PI6,
        success_threshold=0.05,
    ),
    # ─ Peg-Insert ──────────────────────────────────────────────────────
    "peg_insert_side": MetaworldTaskSpec(
        obj_init=(0.0, 0.65, 0.04),
        goal=(-0.35, 0.554, 0.13),
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PEG_TARGET_RADIUS,
    ),
    # ─ Cube + obstacle (push/reach/pick-place variants) ────────────────
    "push_back": MetaworldTaskSpec(
        obj_init=(0.0, 0.85, 0.02),
        goal=(0.0, 0.6, 0.02),
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PUSH_TARGET_RADIUS,
    ),
    "push_wall": MetaworldTaskSpec(
        obj_init=(-0.1, 0.65, 0.02),
        goal=(0.1, 0.85, 0.02),  # diagonal around the wall at y=0.75
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PUSH_TARGET_RADIUS,
    ),
    "reach_wall": MetaworldTaskSpec(
        obj_init=(0.0, 0.65, 0.02),  # cube anchor — not the manipulandum for reach
        goal=(0.0, 0.85, 0.20),  # TCP goal above the wall
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=REACH_TARGET_RADIUS,
    ),
    "pick_place_wall": MetaworldTaskSpec(
        obj_init=(-0.1, 0.65, 0.02),
        goal=(0.1, 0.85, 0.20),  # over the wall, lifted
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PICK_PLACE_SUCCESS_RADIUS,
    ),
    # ─ Cube on a kinematic destination (basketball / shelf / soccer / sweep-into) ─
    "basketball": MetaworldTaskSpec(
        obj_init=(0.0, 0.65, 0.02),
        goal=(0.0, 0.85, 0.25),  # hoop center (basket asset built at z=0.25 when rim_z=0.20 + base z=0.05)
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PICK_PLACE_SUCCESS_RADIUS,
    ),
    "shelf_place": MetaworldTaskSpec(
        obj_init=(0.0, 0.65, 0.02),
        goal=(0.0, 0.85, 0.21),  # shelf top surface
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PICK_PLACE_SUCCESS_RADIUS,
    ),
    "soccer": MetaworldTaskSpec(
        obj_init=(0.0, 0.65, 0.02),
        goal=(0.0, 0.87, 0.06),  # inside the goal volume
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PUSH_TARGET_RADIUS,
    ),
    "sweep": MetaworldTaskSpec(
        obj_init=(0.0, 0.7, 0.02),
        goal=(0.35, 0.65, 0.02),  # off the +x side of the table
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PUSH_TARGET_RADIUS,
    ),
    "sweep_into": MetaworldTaskSpec(
        obj_init=(-0.1, 0.65, 0.02),
        goal=(0.0, 0.75, 0.045),  # bin top center
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PUSH_TARGET_RADIUS,
    ),
    # ─ Plate-slide (articulated; plate is the manipulandum) ────────────
    #
    # Asset internals: groove_base at world (0, 0.75, 0.02); plate_link at
    # initial offset (-0.10, 0, 0.005) from groove_base; prismatic axis X
    # range ``[-0.20, +0.20]``. Plate world-x = ``-0.10 + joint_value``.
    "plate_slide": MetaworldTaskSpec(
        obj_init=(-0.10, 0.75, 0.025),  # joint=0  → plate at world (-0.10, 0.75, 0.025)
        goal=(0.10, 0.75, 0.025),  # joint=+0.20 → plate at world (+0.10, 0.75, 0.025)
        joint_name="plate_slide",
        joint_reset_value=0.0,
        success_threshold=0.05,
    ),
    "plate_slide_back": MetaworldTaskSpec(
        obj_init=(0.10, 0.75, 0.025),  # joint=+0.20 → plate at world (+0.10, 0.75, 0.025)
        goal=(-0.10, 0.75, 0.025),  # joint=0    → plate at world (-0.10, 0.75, 0.025)
        joint_name="plate_slide",
        joint_reset_value=0.20,
        success_threshold=0.05,
    ),
    # ─ Handle press / pull (top-down; prismatic Z, range [-0.10, +0.10]) ─
    #
    # Asset internals: handle_base at world (0, 0.85, 0.05); handle_link at
    # world (0, 0.85, 0.16) at joint=0. Handle marker world-z = ``0.16 + joint``.
    "handle_press": MetaworldTaskSpec(
        obj_init=(0.0, 0.85, 0.21),  # joint=+0.05 → marker at z=0.21 (top)
        goal=(0.0, 0.85, 0.11),  # joint=-0.05 → marker at z=0.11 (bottom)
        joint_name="handle_slide",
        joint_reset_value=0.05,
        success_threshold=0.04,
    ),
    "handle_pull": MetaworldTaskSpec(
        obj_init=(0.0, 0.85, 0.11),  # joint=-0.05
        goal=(0.0, 0.85, 0.21),  # joint=+0.05
        joint_name="handle_slide",
        joint_reset_value=-0.05,
        success_threshold=0.04,
    ),
    # ─ Handle press-side / pull-side (prismatic X, range [-0.10, +0.10]) ──
    #
    # Asset internals: handle_base at (0, 0.85, 0.10); handle_link at world
    # (0.10, 0.85, 0.10) at joint=0. Handle tip marker world-x = ``0.10 + joint``.
    "handle_press_side": MetaworldTaskSpec(
        obj_init=(0.15, 0.85, 0.10),  # joint=+0.05 → marker at x=0.15 (extended)
        goal=(0.05, 0.85, 0.10),  # joint=-0.05 → marker at x=0.05 (pushed in)
        joint_name="handle_slide",
        joint_reset_value=0.05,
        success_threshold=0.04,
    ),
    "handle_pull_side": MetaworldTaskSpec(
        obj_init=(0.05, 0.85, 0.10),
        goal=(0.15, 0.85, 0.10),
        joint_name="handle_slide",
        joint_reset_value=-0.05,
        success_threshold=0.04,
    ),
    # ─ Peg-unplug-side (prismatic X, range [-0.10, 0]) ─────────────────
    #
    # Asset internals: wall_base at (0, 0.85, 0.10); peg_link at (-0.05, 0.85, 0.10)
    # at joint=0; peg_tip marker at (-0.08, 0.85, 0.10). Tip world-x = ``-0.08 + joint``.
    "peg_unplug_side": MetaworldTaskSpec(
        obj_init=(-0.08, 0.85, 0.10),  # joint=0  → fully inserted
        goal=(-0.18, 0.85, 0.10),  # joint=-0.10 → fully pulled out
        joint_name="peg_slide",
        joint_reset_value=0.0,
        success_threshold=0.04,
    ),
    # ─ Box-close (revolute X, hinge at rear edge, range [0°, 120°]) ────
    #
    # Asset internals: box_base at (0, 0.75, 0.04); lid_link at (0, 0.75, 0.085)
    # at joint=0; hinge at rear edge (y=-0.07, z=+0.045 from base center).
    # When joint=θ, lid origin = hinge + R_x(θ) · (0, +0.07, +0.040).
    # At joint=1.5 rad (~85°): marker world ≈ (0, 0.645, 0.118).
    # At joint=0:               marker world  = (0, 0.75,  0.085).
    "box_close": MetaworldTaskSpec(
        obj_init=(0.0, 0.645, 0.118),  # lid open ~85° (joint=1.5 rad)
        goal=(0.0, 0.75, 0.085),  # lid closed (joint=0)
        joint_name="lid_hinge",
        joint_reset_value=1.5,
        success_threshold=0.10,
    ),
    # ─ Hand-insert (TCP target inside a pocket; cube is anchor only) ───
    #
    # The kinematic block has a 5×5 cm pocket centered at world (0, 0.75, 0.04);
    # pocket marker is 2.5 cm below the block top.
    "hand_insert": MetaworldTaskSpec(
        obj_init=(0.0, 0.75, 0.10),  # cube anchor — not the manipulandum
        goal=(0.0, 0.75, 0.015),  # TCP goal inside the pocket
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=REACH_TARGET_RADIUS,
    ),
    # ─ Pick-out-of-hole (cube starts inside the pocket; agent lifts out) ─
    "pick_out_of_hole": MetaworldTaskSpec(
        obj_init=(0.0, 0.75, 0.015),  # cube starts in the pocket
        goal=(0.0, 0.6, 0.20),  # cube lifted out and brought toward agent
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PICK_PLACE_SUCCESS_RADIUS,
    ),
    # ─ Button-press-topdown-wall (button + wall obstacle) ──────────────
    "button_press_topdown_wall": MetaworldTaskSpec(
        obj_init=(0.0, 0.85, 0.13),  # button extended (same as button-press-topdown)
        goal=(0.0, 0.85, 0.07),  # button pressed
        joint_name="btnbox_joint",
        joint_reset_value=0.0,
        success_threshold=BUTTON_TARGET_RADIUS,
    ),
    # ─ Bin-picking (cube into bin; pick-place reward) ──────────────────
    "bin_picking": MetaworldTaskSpec(
        obj_init=(-0.10, 0.65, 0.02),
        goal=(0.0, 0.75, 0.045),  # inside bin (top of walls at z=0.05)
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PICK_PLACE_SUCCESS_RADIUS,
    ),
    # ─ Coffee push/pull (cube as mug-surrogate; coffee_button decoration) ─
    "coffee_push": MetaworldTaskSpec(
        obj_init=(0.0, 0.65, 0.02),
        goal=(0.0, 0.85, 0.02),  # under the coffee machine button
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PUSH_TARGET_RADIUS,
    ),
    "coffee_pull": MetaworldTaskSpec(
        obj_init=(0.0, 0.85, 0.02),
        goal=(0.0, 0.65, 0.02),  # toward the agent
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PUSH_TARGET_RADIUS,
    ),
    # ─ Stick push/pull (cube + stick decoration; cube IS the manipulandum) ─
    "stick_push": MetaworldTaskSpec(
        obj_init=(0.0, 0.65, 0.02),
        goal=(0.20, 0.65, 0.02),  # far +x
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PUSH_TARGET_RADIUS,
    ),
    "stick_pull": MetaworldTaskSpec(
        obj_init=(0.20, 0.65, 0.02),
        goal=(0.0, 0.65, 0.02),
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PUSH_TARGET_RADIUS,
    ),
    # ─ Assembly / Disassemble (cube as ring-surrogate; assembly_peg) ───
    #
    # Asset internals: assembly_peg base at (0, 0.75, 0.025); peg_tip
    # marker at (0, 0.75, 0.110) (base + 0.085).
    "assembly": MetaworldTaskSpec(
        obj_init=(0.0, 0.55, 0.02),  # cube on table near agent
        goal=(0.0, 0.75, 0.110),  # placed on top of peg
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PICK_PLACE_SUCCESS_RADIUS,
    ),
    "disassemble": MetaworldTaskSpec(
        obj_init=(0.0, 0.75, 0.110),  # cube starts on peg tip
        goal=(0.0, 0.55, 0.20),  # lifted off and brought toward agent
        joint_name=None,
        joint_reset_value=0.0,
        success_threshold=PICK_PLACE_SUCCESS_RADIUS,
    ),
    # ─ Hammer (nail is the articulated manipulandum) ───────────────────
    #
    # Asset internals: block_base at (0, 0.80, 0.04); nail_head marker world-z
    # = 0.10 + 0.02 + joint = 0.12 + joint. Range [-0.06, 0]: extended at
    # joint=0 (z=0.12), driven flush at joint=-0.06 (z=0.06).
    "hammer": MetaworldTaskSpec(
        obj_init=(0.0, 0.80, 0.12),  # nail extended
        goal=(0.0, 0.80, 0.06),  # nail driven flush
        joint_name="nail_drive",
        joint_reset_value=0.0,
        success_threshold=0.04,
    ),
    # ─ Plate-slide-side / -back-side (plate slides along world Y) ──────
    #
    # Asset internals (mw_plate_side.usda): groove_base at (0, 0.75, 0.02);
    # plate_link at world (0, 0.65, 0.025) at joint=0; prismatic axis Y range
    # [-0.20, +0.20]. Plate world-y = ``0.65 + joint``.
    "plate_slide_side": MetaworldTaskSpec(
        obj_init=(0.0, 0.65, 0.025),  # joint=0    → plate at -y end
        goal=(0.0, 0.85, 0.025),  # joint=+0.20 → plate at +y end
        joint_name="plate_slide",
        joint_reset_value=0.0,
        success_threshold=0.05,
    ),
    "plate_slide_back_side": MetaworldTaskSpec(
        obj_init=(0.0, 0.85, 0.025),  # joint=+0.20
        goal=(0.0, 0.65, 0.025),  # joint=0
        joint_name="plate_slide",
        joint_reset_value=0.20,
        success_threshold=0.05,
    ),
    # ─ Button-press / button-press-wall (front-facing button, prismatic Y) ─
    #
    # Asset internals (mw_button_front.usda): button_box at (0, 0.85, 0.10);
    # button_link at world (0, 0.79, 0.10) at joint=0; range [0, +0.06].
    # Marker world-y = ``0.79 + joint``: extended at joint=0 (y=0.79),
    # pressed at joint=+0.06 (y=0.85, retracted into mount).
    "button_press": MetaworldTaskSpec(
        obj_init=(0.0, 0.79, 0.10),  # joint=0  → cap protruded
        goal=(0.0, 0.85, 0.10),  # joint=+0.06 → cap pressed in
        joint_name="btnbox_joint",
        joint_reset_value=0.0,
        success_threshold=BUTTON_TARGET_RADIUS,
    ),
    "button_press_wall": MetaworldTaskSpec(
        obj_init=(0.0, 0.79, 0.10),
        goal=(0.0, 0.85, 0.10),
        joint_name="btnbox_joint",
        joint_reset_value=0.0,
        success_threshold=BUTTON_TARGET_RADIUS,
    ),
}
"""All real-asset task specs. Keys must match the gym IDs used in
``config/sawyer/__init__.py`` (with ``-`` → ``_``)."""


def _merge_mw_ranges() -> None:
    """Merge MW-derived sampling ranges into :data:`TASK_SPECS` in-place.

    The ranges come from a frozen MT50 dump (`utils/parity/mw_ranges_baked.py`)
    so single-task and multi-task envs sample obj_init / goal from the same
    per-episode distribution Meta-World uses. Tasks not present in the
    baked table keep point sampling (degenerate range)."""

    import dataclasses

    from .utils.parity.mw_ranges_baked import MW_RANGES

    for spec_key, s in list(TASK_SPECS.items()):
        mw_name = spec_key.replace("_", "-") + "-v3"
        r = MW_RANGES.get(mw_name)
        if r is None:
            continue
        TASK_SPECS[spec_key] = dataclasses.replace(
            s,
            obj_range_low=r["obj_low"],
            obj_range_high=r["obj_high"],
            goal_range_low=r["goal_low"],
            goal_range_high=r["goal_high"],
        )


_merge_mw_ranges()


__all__ = ["MetaworldTaskSpec", "TASK_SPECS"]
