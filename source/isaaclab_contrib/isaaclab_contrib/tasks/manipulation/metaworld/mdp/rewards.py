# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success indicators + per-task target radius constants for Meta-World tasks.

Per-task reward composition lives entirely in env-cfgs, expressed as
:class:`RewardTermCfg` against the four archetypes in :mod:`reward_shapes`:

* :class:`ToleranceShapeCfg` → :func:`tolerance_shape`
* :class:`HamacherShapeCfg` → :func:`hamacher_shape`
* :class:`CagingTimesInPlaceShapeCfg` → :func:`caging_times_in_place_shape`
* :class:`LinearComboShapeCfg` → :func:`linear_combo_shape`

There are NO task-named reward functions in this module. If you find
yourself writing ``def <task>_v2(...)``, that's a code smell — instead
compose archetypes in the task's env-cfg.

This module exposes:
* The MW per-task target-radius constants (used by both the env-cfg
  archetype params and the success indicator).
* :func:`reach_success` — TCP-to-goal binary indicator (MT3 reach).
* :func:`keypoint_at_target` — generic manipulandum-to-goal binary
  indicator that works for every non-reach task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from . import quantities as q

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Task-radius constants from MW source ───────────────────────────────────

REACH_TARGET_RADIUS: float = 0.05
PUSH_TARGET_RADIUS: float = 0.05
PICK_PLACE_TARGET_RADIUS: float = 0.05
PICK_PLACE_SUCCESS_RADIUS: float = 0.07
DRAWER_TARGET_RADIUS: float = 0.04
WINDOW_TARGET_RADIUS: float = 0.05
DOOR_TARGET_RADIUS: float = 0.08
BUTTON_TARGET_RADIUS: float = 0.024
PEG_TARGET_RADIUS: float = 0.07


# ── Success indicators ─────────────────────────────────────────────────────


def reach_success(
    env: ManagerBasedRLEnv,
    *,
    frame_transformer_cfg: SceneEntityCfg = SceneEntityCfg("tcp_frame"),
    goal_command_name: str = "ee_pose",
    threshold: float = REACH_TARGET_RADIUS,
) -> torch.Tensor:
    """1.0 when TCP is within ``threshold`` m of the goal. Used only by reach;
    every other task uses :func:`keypoint_at_target`."""
    return (
        q.tcp_to_target_dist(env, frame_transformer_cfg=frame_transformer_cfg, goal_command_name=goal_command_name)
        <= threshold
    ).to(torch.float32)


def keypoint_at_target(
    env: ManagerBasedRLEnv,
    *,
    keypoint_frame_cfg: SceneEntityCfg = SceneEntityCfg("keypoint_frame"),
    goal_command_name: str = "ee_pose",
    threshold: float = 0.05,
) -> torch.Tensor:
    """1.0 when the manipulandum keypoint is within ``threshold`` m of the goal."""
    return (
        q.obj_to_target_dist(env, keypoint_frame_cfg=keypoint_frame_cfg, goal_command_name=goal_command_name)
        <= threshold
    ).to(torch.float32)
