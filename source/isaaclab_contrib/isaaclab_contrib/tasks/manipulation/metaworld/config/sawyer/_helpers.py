# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared cfg-builder helpers for Sawyer Meta-World envs.

Pulled out of ``env_cfgs.py`` so that single-task and multi-task cfgs can
import the same primitives without env_cfgs.py becoming a 2000-line catch-
all. Contains:

* :func:`_fixed_paired` — degenerate-range :class:`MetaworldPairedCommandCfg`.
* :func:`_paired_from_spec` — same, but reading sampling ranges from
  :data:`TASK_SPECS`.
* :func:`_reset_robot` — Sawyer joint reset event term (small ±0.05 rad noise).
* :func:`_reset_joint_to` — generic ``cabinet`` joint reset to a fixed value.
* :func:`_reset_joint_from_spec` — same, but reading the joint name + value
  from :data:`TASK_SPECS`.

Per-archetype reward shape factories (``_DRAWER_OPEN_*``, ``_PEG_INSERT_*``,
etc.) still live in ``env_cfgs.py`` for now — they are tightly coupled to
the per-task ``RewardTermCfg`` declarations. Phase 7 of the refactor plan
extracts those next.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg, SceneEntityCfg

from ...mdp import MetaworldPairedCommandCfg
from ...metaworld_specs import TASK_SPECS


def _fixed_paired(obj, goal) -> MetaworldPairedCommandCfg:
    """Build a degenerate-range :class:`MetaworldPairedCommandCfg`
    (``low == high == obj/goal``) — the same point every episode."""
    return MetaworldPairedCommandCfg(
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        object_name="cube",
        frame_transformer_name="tcp_frame",
        obj_low=obj,
        obj_high=obj,
        goal_low=goal,
        goal_high=goal,
        min_xy_separation=0.0,
    )


def _paired_from_spec(task_name: str) -> MetaworldPairedCommandCfg:
    """``MetaworldPairedCommandCfg`` reading sampling ranges from
    :data:`TASK_SPECS`. Falls back to point sampling when a task has no
    ``*_range_*`` populated yet (preserves pre-parity behaviour for unaudited
    tasks)."""
    spec = TASK_SPECS[task_name]
    obj_low, obj_high = spec.obj_range()
    goal_low, goal_high = spec.goal_range()
    return MetaworldPairedCommandCfg(
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        object_name="cube",
        frame_transformer_name="tcp_frame",
        obj_low=obj_low,
        obj_high=obj_high,
        goal_low=goal_low,
        goal_high=goal_high,
        min_xy_separation=0.0,
    )


def _reset_robot() -> EventTermCfg:
    """Reset Sawyer joints by a small ±0.05 rad offset every episode."""
    return EventTermCfg(
        func="isaaclab.envs.mdp:reset_joints_by_offset",
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )


def _reset_joint_to(joint_name: str, value: float) -> EventTermCfg:
    """Reset the asset's articulated joint to a fixed value (no noise)."""
    return EventTermCfg(
        func="isaaclab.envs.mdp:reset_joints_by_offset",
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=[joint_name]),
            "position_range": (value, value),
            "velocity_range": (0.0, 0.0),
        },
    )


def _reset_joint_from_spec(task_name: str) -> EventTermCfg:
    """``_reset_joint_to`` reading ``joint_name`` and ``joint_reset_value`` from
    :data:`TASK_SPECS`."""
    spec = TASK_SPECS[task_name]
    if spec.joint_name is None:
        raise ValueError(f"Task {task_name!r} has no asset joint to reset.")
    return _reset_joint_to(spec.joint_name, spec.joint_reset_value)


__all__ = [
    "_fixed_paired",
    "_paired_from_spec",
    "_reset_robot",
    "_reset_joint_to",
    "_reset_joint_from_spec",
]
