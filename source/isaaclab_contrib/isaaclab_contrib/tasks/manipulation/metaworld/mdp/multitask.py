# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Heterogeneous multi-task primitives for Meta-World+.

Three pieces work together:

* :class:`MetaworldMultiTaskCommand` extends :class:`MetaworldPairedCommand`
  with per-task sample boxes — each env's task is assigned at construction
  (round-robin across the registered task list), and each env samples its
  ``(obj_init, goal)`` from its assigned task's box. Exposes ``self.task_id``
  (LongTensor, shape ``(num_envs,)``) for downstream consumers.
* :func:`metaworld_task_onehot` returns one-hot of each env's task_id, used
  as an observation term so the policy knows which task it's on.
* :func:`task_masked_reward` wraps any single-task reward function so it
  contributes only on envs whose ``task_id`` matches a given index.

Together these replace what would otherwise be ``N`` separate paired-command
+ reward + obs setups (one per task) with a single set of cfgs that route
per-env behaviour by ``task_id``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import MISSING, field
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.utils import configclass

from .commands import MetaworldPairedCommand, MetaworldPairedCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Per-task sample-box cfg ─────────────────────────────────────────────────


@configclass
class TaskBoxCfg:
    """One task's spawn distribution + label."""

    name: str = MISSING
    """Short task name (e.g. ``"reach"``, ``"push"``). Used for logging."""

    obj_low: tuple[float, float, float] = MISSING
    obj_high: tuple[float, float, float] = MISSING
    goal_low: tuple[float, float, float] = MISSING
    goal_high: tuple[float, float, float] = MISSING


# ── Multi-task command term ─────────────────────────────────────────────────


@configclass
class MetaworldMultiTaskCommandCfg(MetaworldPairedCommandCfg):
    """Cfg for the multi-task paired command.

    Note:
        ``obj_low``/``obj_high``/``goal_low``/``goal_high`` from the parent
        cfg are ignored — sample boxes are read from :attr:`tasks` per env.
        We supply harmless defaults so the parent's ``MISSING`` checks pass.
    """

    class_type: type | str = "isaaclab_contrib.tasks.manipulation.metaworld.mdp.multitask:MetaworldMultiTaskCommand"

    tasks: list[TaskBoxCfg] = field(default_factory=list)
    """Task spec list. Env ``i`` is assigned task ``i % len(tasks)``."""

    obj_low: tuple[float, float, float] = (0.0, 0.0, 0.0)
    obj_high: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_low: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_high: tuple[float, float, float] = (0.0, 0.0, 0.0)


class MetaworldMultiTaskCommand(MetaworldPairedCommand):
    """Per-env-task paired command. Samples each env from its assigned task.

    Exposes :attr:`task_id` (LongTensor, shape ``(num_envs,)``) for reward
    terms to mask their contribution.
    """

    cfg: MetaworldMultiTaskCommandCfg

    def __init__(self, cfg: MetaworldMultiTaskCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        device = env.device

        if not cfg.tasks:
            raise ValueError("MetaworldMultiTaskCommandCfg.tasks must be non-empty")
        n_tasks = len(cfg.tasks)
        # Round-robin assignment: env i → task i % n_tasks.
        self.task_id: torch.Tensor = torch.arange(env.num_envs, device=device, dtype=torch.long) % n_tasks
        self._n_tasks: int = n_tasks
        self._task_obj_low = torch.stack([torch.tensor(t.obj_low, device=device) for t in cfg.tasks])
        self._task_obj_high = torch.stack([torch.tensor(t.obj_high, device=device) for t in cfg.tasks])
        self._task_goal_low = torch.stack([torch.tensor(t.goal_low, device=device) for t in cfg.tasks])
        self._task_goal_high = torch.stack([torch.tensor(t.goal_high, device=device) for t in cfg.tasks])

    @property
    def num_tasks(self) -> int:
        return self._n_tasks

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Sample (obj, goal) per env using its assigned task's box."""
        ids = (
            env_ids
            if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(list(env_ids), device=self._env.device, dtype=torch.long)
        )
        if ids.numel() == 0:
            return

        device = self._env.device
        n = ids.numel()
        tids = self.task_id[ids]  # (n,) task index per env

        # Per-env low/high vectors.
        obj_low = self._task_obj_low[tids]  # (n, 3)
        obj_high = self._task_obj_high[tids]
        goal_low = self._task_goal_low[tids]
        goal_high = self._task_goal_high[tids]

        obj = obj_low + (obj_high - obj_low) * torch.rand((n, 3), device=device)
        goal = goal_low + (goal_high - goal_low) * torch.rand((n, 3), device=device)

        # Vectorised rejection on xy separation.
        for _ in range(self.cfg.max_resample_iters):
            xy_dist = torch.linalg.norm(obj[:, :2] - goal[:, :2], dim=-1)
            bad = xy_dist < self.cfg.min_xy_separation
            if not bad.any():
                break
            n_bad = int(bad.sum().item())
            obj_re = obj_low[bad] + (obj_high[bad] - obj_low[bad]) * torch.rand((n_bad, 3), device=device)
            goal_re = goal_low[bad] + (goal_high[bad] - goal_low[bad]) * torch.rand((n_bad, 3), device=device)
            obj[bad] = obj_re
            goal[bad] = goal_re

        # Write goal + obj init buffers (parent class handles reset state).
        self._command_buf[ids] = goal
        self.obj_init_pos_e[ids] = obj

        # Apply the obj pose in world frame.
        env_origins = self._env.scene.env_origins[ids]
        obj_pos_w = env_origins + obj
        quat = torch.zeros((n, 4), device=device)
        quat[:, 0] = 1.0
        pose = torch.cat([obj_pos_w, quat], dim=-1)
        self._object.write_root_pose_to_sim(pose, env_ids=ids)
        self._object.write_root_velocity_to_sim(torch.zeros((n, 6), device=device), env_ids=ids)


# ── Observation + reward helpers that read ``task_id`` ──────────────────────


def metaworld_task_onehot(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Returns ``(num_envs, num_tasks)`` one-hot of the per-env task assignment.

    Args:
        env: The active environment.
        command_name: Name of the :class:`MetaworldMultiTaskCommand` term in
            ``env.command_manager``.

    Returns:
        ``one_hot`` (float32) shape ``(num_envs, num_tasks)``.
    """
    cmd: MetaworldMultiTaskCommand = env.command_manager.get_term(command_name)
    n_tasks = cmd.num_tasks
    return torch.nn.functional.one_hot(cmd.task_id, num_classes=n_tasks).to(torch.float32)


def task_masked_reward(
    env: ManagerBasedRLEnv,
    *,
    inner_func: Callable[..., torch.Tensor] | str,
    inner_kwargs: dict[str, Any],
    task_index: int,
    command_name: str = "ee_pose",
) -> torch.Tensor:
    """Compute ``inner_func(env, **inner_kwargs)`` but zero it on envs whose
    ``task_id != task_index``.

    .. deprecated::
        Use the ``@scatterable`` atoms in
        :mod:`~isaaclab_contrib.tasks.manipulation.metaworld.mdp.scatter_rewards`
        with ``asset_cfg=SceneEntityCfg(<keypoint>, groups=[task_name])``
        instead. This wrapper is no longer used by any multi-task env cfg
        and will be removed in a future release.

    Args:
        env: The active environment.
        inner_func: Reward function (or import path string) producing a
            ``(num_envs,)`` tensor.
        inner_kwargs: Kwargs forwarded to ``inner_func``.
        task_index: Index of the task this reward applies to.
        command_name: Name of the multi-task command term.

    Returns:
        ``(num_envs,)`` reward, zero outside the assigned task.
    """
    # Lazy import-by-string support so YAML/configclass paths work.
    if isinstance(inner_func, str):
        mod_path, _, attr = inner_func.rpartition(":")
        if not mod_path:
            mod_path, _, attr = inner_func.rpartition(".")
        import importlib

        inner_func = getattr(importlib.import_module(mod_path), attr)

    raw = inner_func(env, **inner_kwargs)  # (N,) or (N, 1)
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw.squeeze(-1)

    cmd: MetaworldMultiTaskCommand = env.command_manager.get_term(command_name)
    mask = (cmd.task_id == task_index).to(raw.dtype)
    return raw * mask
