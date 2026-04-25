# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal velocity-tracking command term — a debug control for :class:`MultiTaskCommand`.

Drop-in replacement for ``MultiTaskCommand`` on the pure-velocity-tracking task.
Same interface — ``.command``, ``.task_reward``, ``.task_done`` — so the env cfg
can swap the implementation without touching reward / termination / obs wiring.
Same sparse terminal reward structure (``G = mean_t A_t`` over the episode,
emitted only on timeout; zero otherwise) so the **reward signal magnitude and
density are identical** to our multi-task composer on the ``velocity`` preset.

If this trains and the multi-task composer on the same task does not, the bug
is inside :class:`MultiTaskCommand`'s dispatch or spec machinery, not in the
sparse-reward formulation. The math here is deliberately trivial: no Warp
dispatch, no canonical layouts, no per-slot spec tables, no kernel enums.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MinimalVelocityCommand(CommandTerm):
    """Sparse-terminal velocity tracker — linear-xy + angular-z.

    Per-env target: sample once per episode (on reset) uniformly from
    :attr:`MinimalVelocityCommandCfg.lin_vel_range` and ``ang_vel_range``.

    Per-step: compute tanh-kernel activation from velocity error, accumulate.

    At the last step of the episode (timeout): emit
    ``task_reward = mean(A_t over all steps) ∈ [0, 1]``. Otherwise emit 0.

    This is what our multi-task composer computes for a pure-tracking task,
    minus the abstraction layers. Identical reward density (one non-zero
    emission per episode), identical reward magnitude (mean activation ∈ [0,1]).
    """

    cfg: MinimalVelocityCommandCfg

    def __init__(self, cfg: MinimalVelocityCommandCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._env = env
        self._robot = env.scene[cfg.asset_cfg.name]
        self._max_episode_length = int(env.max_episode_length)
        device = self.device
        num_envs = self.num_envs

        # Per-env target: (lin_x, lin_y, ang_z). Shape [num_envs, 3].
        self._target = torch.zeros((num_envs, 3), device=device)
        # Accumulators cleared on reset.
        self._sum_activation = torch.zeros(num_envs, device=device)
        self._transit_steps = torch.zeros(num_envs, dtype=torch.int32, device=device)
        # Outputs (refreshed every step).
        self._task_reward = torch.zeros(num_envs, device=device)
        self._task_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Obs compatibility: mirror ``MultiTaskCommand``'s interface so the env cfg
        # can keep the same ObsTerms (``command_reach``, ``command_track``,
        # ``command_active``, ``command_progress``). No instant subtasks here, so
        # ``command_reach`` is empty; ``command_track`` carries the 3-vec velocity
        # delta; ``command_active`` is all-ones; ``progress`` is the current step's
        # mean activation.
        self._command_reach = torch.zeros((num_envs, 0), device=device)
        self._command_track = torch.zeros((num_envs, 3), device=device)
        self._command_active = torch.ones((num_envs, 3), device=device)
        self._progress = torch.zeros(num_envs, device=device)

    # ------------------------------------------------------------------------
    # Required CommandTerm API.
    # ------------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """Target exposed to the policy as the ``task`` obs. Shape ``[num_envs, 3]``."""
        return self._target

    @property
    def task_reward(self) -> torch.Tensor:
        return self._task_reward

    @property
    def task_done(self) -> torch.Tensor:
        return self._task_done

    # --- obs-compat properties (mirror :class:`MultiTaskCommand` interface) ---

    @property
    def progress(self) -> torch.Tensor:
        return self._progress

    @property
    def command_reach(self) -> torch.Tensor:
        return self._command_reach

    @property
    def command_track(self) -> torch.Tensor:
        return self._command_track

    @property
    def command_active(self) -> torch.Tensor:
        return self._command_active

    def _update_metrics(self) -> None:
        self.metrics["error/body_lin_vel"] = self._lin_vel_error_xy()
        self.metrics["error/body_ang_vel"] = self._ang_vel_error_z()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        lo_lin, hi_lin = self.cfg.lin_vel_range
        lo_ang, hi_ang = self.cfg.ang_vel_range
        self._target[env_ids, 0].uniform_(lo_lin, hi_lin)
        self._target[env_ids, 1].uniform_(lo_lin, hi_lin)
        self._target[env_ids, 2].uniform_(lo_ang, hi_ang)
        # Clear accumulators — new episode starts fresh.
        self._sum_activation[env_ids] = 0.0
        self._transit_steps[env_ids] = 0
        self._task_reward[env_ids] = 0.0
        self._task_done[env_ids] = False

    def _update_command(self) -> None:
        """Per-step: accumulate activation, emit terminal reward on timeout."""
        err_lin = self._lin_vel_error_xy()
        err_ang = self._ang_vel_error_z()
        # Mean of two tanh activations — same shape as multi-task composer's
        # ``tracking_mean`` for the velocity preset (two tracking subtasks).
        act_lin = 1.0 - torch.tanh(err_lin / self.cfg.activation_std)
        act_ang = 1.0 - torch.tanh(err_ang / self.cfg.activation_std)
        per_step_activation = 0.5 * (act_lin + act_ang)

        self._sum_activation.add_(per_step_activation)
        self._transit_steps.add_(1)

        # Latch terminal reward one step before the outer ``time_out`` DoneTerm
        # fires — matches the ``is_timeout = buf >= max - 1`` convention used by
        # ``MultiTaskCommand`` so the timing is apples-to-apples.
        is_timeout = self._env.episode_length_buf >= self._max_episode_length - 1
        transit_mean = self._sum_activation / self._transit_steps.clamp(min=1).to(self._sum_activation.dtype)
        self._task_reward = torch.where(is_timeout, transit_mean, torch.zeros_like(transit_mean))
        # ``task_done`` stays False — episodes end on timeout / base_contact,
        # not on command-success (no instant subtask to achieve).
        self._task_done.zero_()

        # Obs mirrors.
        cur_lin = wp.to_torch(self._robot.data.root_lin_vel_b)[:, :2]
        cur_ang = wp.to_torch(self._robot.data.root_ang_vel_b)[:, 2]
        self._command_track[:, :2] = self._target[:, :2] - cur_lin
        self._command_track[:, 2] = self._target[:, 2] - cur_ang
        self._progress.copy_(per_step_activation)

    # ------------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------------

    def _lin_vel_error_xy(self) -> torch.Tensor:
        # Articulation data are Warp arrays — convert via zero-copy torch view.
        cur = wp.to_torch(self._robot.data.root_lin_vel_b)[:, :2]  # base-frame xy
        tgt = self._target[:, :2]
        return torch.linalg.vector_norm(cur - tgt, dim=-1)

    def _ang_vel_error_z(self) -> torch.Tensor:
        cur = wp.to_torch(self._robot.data.root_ang_vel_b)[:, 2]  # base-frame z
        tgt = self._target[:, 2]
        return (cur - tgt).abs()


@configclass
class MinimalVelocityCommandCfg(CommandTermCfg):
    """Cfg for :class:`MinimalVelocityCommand` — a debug control for the composer."""

    class_type: type = MinimalVelocityCommand
    asset_cfg: SceneEntityCfg = MISSING  # type: ignore[assignment]
    lin_vel_range: tuple[float, float] = (-1.0, 1.0)
    """Uniform xy lin-vel target range [m/s]."""
    ang_vel_range: tuple[float, float] = (-1.0, 1.0)
    """Uniform z ang-vel target range [rad/s]."""
    activation_std: float = 1.0
    """``A = 1 - tanh(err / activation_std)``. Match the multi-task preset for apples-to-apples."""
