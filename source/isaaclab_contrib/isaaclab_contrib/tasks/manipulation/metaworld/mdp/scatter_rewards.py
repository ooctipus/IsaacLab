# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Group-scoped (``@scatterable``) reward atoms for the Meta-World port.

The existing reward primitives in :mod:`reward_shapes` and
:mod:`quantities` compute for *all* envs. In a multi-task env where tasks
are dispatched via clone groups, we want each task's reward to fire only
on its own envs. The ``@scatterable`` wrappers here:

1. Accept a :class:`SceneEntityCfg` with ``groups=[task_name]``.
2. Run the existing primitive (which still computes for all envs).
3. Index the result by ``cfg.env_ids`` and let the
   :func:`~...multitask.mdp.utils.scatterable` decorator scatter into a
   ``(num_envs,)`` buffer with zeros on non-matching envs.

Per-task reward decomposition stays explicit at the ``RewardTermCfg``
declaration site — typically two small terms (monotonic shaped reward +
success indicator) per task. No ``task_masked_reward``, no per-task
wrapper functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...multitask.mdp.utils import ScatterResult, scatterable
from .quantities import obj_to_target_dist, tcp_to_target_dist
from .reward_shapes import (
    CagingTimesInPlaceShapeCfg,
    HamacherShapeCfg,
    LinearComboShapeCfg,
    ToleranceShapeCfg,
    caging_times_in_place_shape,
    hamacher_shape,
    linear_combo_shape,
    tolerance_shape,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


@scatterable(output_dim=0)
def linear_combo_term(
    env: ManagerBasedRLEnv,
    *,
    cfg: LinearComboShapeCfg,
    asset_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Group-scoped wrapper around :func:`linear_combo_shape`.

    The inner ``LinearComboShapeCfg`` may include or omit
    :attr:`success_override` — both work. Use the *no-override* form here
    and pair with :func:`success_indicator_term` to keep success as its
    own additive term.
    """
    full = linear_combo_shape(env, cfg=cfg)
    return asset_cfg.env_ids, full[asset_cfg.env_ids]


@scatterable(output_dim=0)
def caging_times_in_place_term(
    env: ManagerBasedRLEnv,
    *,
    cfg: CagingTimesInPlaceShapeCfg,
    asset_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Group-scoped wrapper around :func:`caging_times_in_place_shape`."""
    full = caging_times_in_place_shape(env, cfg=cfg)
    return asset_cfg.env_ids, full[asset_cfg.env_ids]


@scatterable(output_dim=0)
def hamacher_term(
    env: ManagerBasedRLEnv,
    *,
    cfg: HamacherShapeCfg,
    asset_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Group-scoped wrapper around :func:`hamacher_shape`."""
    full = hamacher_shape(env, cfg=cfg)
    return asset_cfg.env_ids, full[asset_cfg.env_ids]


@scatterable(output_dim=0)
def tolerance_term(
    env: ManagerBasedRLEnv,
    *,
    cfg: ToleranceShapeCfg,
    asset_cfg: SceneEntityCfg,
) -> ScatterResult:
    """Group-scoped wrapper around :func:`tolerance_shape`."""
    full = tolerance_shape(env, cfg=cfg)
    return asset_cfg.env_ids, full[asset_cfg.env_ids]


@scatterable(output_dim=0)
def success_indicator_term(
    env: ManagerBasedRLEnv,
    *,
    keypoint_frame_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    goal_command_name: str = "ee_pose",
    threshold: float = 0.05,
) -> ScatterResult:
    """Per-step success bonus: ``1.0`` when ``‖keypoint − goal‖ <= threshold``,
    else ``0``. Scatter to ``asset_cfg.env_ids`` so only matching envs get
    the bonus.

    Pair with a :class:`RewardTermCfg`'s ``weight`` to set the bonus
    magnitude (e.g. ``weight=10.0`` mirrors MW's ``reward = 10.0`` success
    override; here it adds 10 instead of replacing, which still favours
    success over the typical ≈10 reward at the converged state).
    """
    distance = obj_to_target_dist(env, keypoint_frame_cfg=keypoint_frame_cfg, goal_command_name=goal_command_name)
    success = (distance <= threshold).to(torch.float32)
    return asset_cfg.env_ids, success[asset_cfg.env_ids]


@scatterable(output_dim=0)
def reach_success_term(
    env: ManagerBasedRLEnv,
    *,
    frame_transformer_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    goal_command_name: str = "ee_pose",
    threshold: float = 0.05,
) -> ScatterResult:
    """Per-step reach-success bonus: ``1.0`` when ``‖TCP − goal‖ <= threshold``,
    else ``0``. Scatter to ``asset_cfg.env_ids`` so only matching envs get
    the bonus.

    Used by the MT3 reach task — :func:`success_indicator_term` reads the
    object keypoint, but reach's success criterion is TCP-to-goal.
    """
    distance = tcp_to_target_dist(env, frame_transformer_cfg=frame_transformer_cfg, goal_command_name=goal_command_name)
    success = (distance <= threshold).to(torch.float32)
    return asset_cfg.env_ids, success[asset_cfg.env_ids]


@scatterable(output_dim=0, dtype=torch.bool)
def keypoint_success_termination(
    env: ManagerBasedRLEnv,
    *,
    keypoint_frame_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    goal_command_name: str = "ee_pose",
    threshold: float = 0.05,
) -> ScatterResult:
    """Bool variant of :func:`success_indicator_term` for use in
    :class:`~isaaclab.managers.TerminationTermCfg` — returns ``True`` on
    envs where ``‖keypoint − goal‖ <= threshold``, scattered to
    ``asset_cfg.env_ids`` so only matching envs can terminate.

    Mirrors Meta-World V2's done-on-success behaviour.
    """
    distance = obj_to_target_dist(env, keypoint_frame_cfg=keypoint_frame_cfg, goal_command_name=goal_command_name)
    success = distance <= threshold
    return asset_cfg.env_ids, success[asset_cfg.env_ids]


@scatterable(output_dim=0, dtype=torch.bool)
def reach_success_termination(
    env: ManagerBasedRLEnv,
    *,
    frame_transformer_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    goal_command_name: str = "ee_pose",
    threshold: float = 0.05,
) -> ScatterResult:
    """Bool variant of :func:`reach_success_term` — returns ``True`` on
    envs where ``‖TCP − goal‖ <= threshold``, scattered to
    ``asset_cfg.env_ids``. Used by the MT3 reach task whose success
    criterion is TCP-to-goal (not object-to-goal)."""
    distance = tcp_to_target_dist(env, frame_transformer_cfg=frame_transformer_cfg, goal_command_name=goal_command_name)
    success = distance <= threshold
    return asset_cfg.env_ids, success[asset_cfg.env_ids]


__all__ = [
    "caging_times_in_place_term",
    "hamacher_term",
    "keypoint_success_termination",
    "linear_combo_term",
    "reach_success_term",
    "reach_success_termination",
    "success_indicator_term",
    "tolerance_term",
]
