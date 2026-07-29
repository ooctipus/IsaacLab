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
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

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

        # Termination-predicate compatibility: ``time_out_reach_truncate`` and
        # ``time_out_track_terminate`` (declared on :class:`MultiTaskEnvCfg`'s
        # ``MultiTaskTerminationsCfg``) read three :class:`MultiTaskCommand`
        # attributes: ``spec.task_has_instant[task_samples]`` to decide
        # truncate-vs-terminate, and ``effective_max_episode_length`` to honor
        # the adaptive episode-length curriculum. Mock them out as a single
        # pure-tracking task — every env always falls through to the
        # ``time_out_track_terminate`` branch (reach-truncate is False
        # everywhere because no task has an instant subtask).
        self.spec = SimpleNamespace(task_has_instant=torch.zeros(1, dtype=torch.bool, device=device))
        self.task_samples = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Per-env effective episode cap, mirroring :class:`MultiTaskCommand`'s
        # adaptive episode-length curriculum so apples-to-apples comparison is
        # possible. When ``cfg.tracking_episode_length_min_seconds`` is None the
        # cap stays at ``max_episode_length``; when set, every reset draws a
        # fresh uniform length in ``[_random_episode_min_steps,
        # max_episode_length]``.
        self._random_episode_enabled = cfg.tracking_episode_length_min_seconds is not None
        step_dt = float(getattr(env, "step_dt", 0.02))
        min_s = cfg.tracking_episode_length_min_seconds or 0.0
        self._random_episode_min_steps = max(1, int(round(min_s / step_dt)))
        self._effective_max_episode_length = torch.full(
            (num_envs,), self._max_episode_length, dtype=torch.int32, device=device
        )

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

    @property
    def effective_max_episode_length(self) -> torch.Tensor:
        """Per-env episode-end step count [num_envs, int32].

        Mirrors :attr:`MultiTaskCommand.effective_max_episode_length`. When
        :attr:`MinimalVelocityCommandCfg.tracking_episode_length_min_seconds`
        is set, a fresh uniform sample in ``[_random_episode_min_steps,
        max_episode_length]`` is drawn on every reset; otherwise stays at
        ``max_episode_length`` everywhere. Consumed by
        :func:`time_out_track_terminate`.
        """
        return self._effective_max_episode_length

    def _update_metrics(self) -> None:
        self.metrics["error/body_lin_vel"] = self._lin_vel_error()
        self.metrics["error/body_ang_vel"] = self._ang_vel_error()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        lo_lin, hi_lin = self.cfg.lin_vel_range
        lo_ang, hi_ang = self.cfg.ang_vel_range
        # ``self._target[env_ids, j]`` with ``env_ids`` a LongTensor is advanced
        # indexing — it returns a copy, so ``.uniform_()`` on it would no-op
        # against the original. Sample into a fresh ``[n, 3]`` tensor (basic
        # indexing on its column views) and assign the row block back; the
        # row-block ``__setitem__`` is the supported in-place path.
        n = int(env_ids.numel())
        device = self._target.device
        new_target = torch.empty((n, 3), device=device)
        new_target[:, 0].uniform_(lo_lin, hi_lin)
        new_target[:, 1].uniform_(lo_lin, hi_lin)
        new_target[:, 2].uniform_(lo_ang, hi_ang)
        self._target[env_ids] = new_target
        # Clear accumulators — new episode starts fresh.
        self._sum_activation[env_ids] = 0.0
        self._transit_steps[env_ids] = 0
        self._task_reward[env_ids] = 0.0
        self._task_done[env_ids] = False
        # Adaptive episode-length curriculum (mirrors MultiTaskCommand):
        # draw fresh per-env caps when enabled.
        if self._random_episode_enabled:
            self._effective_max_episode_length[env_ids] = torch.randint(
                self._random_episode_min_steps,
                self._max_episode_length + 1,
                (env_ids.numel(),),
                dtype=torch.int32,
                device=self.device,
            )

    def _update_command(self) -> None:
        """Per-step: accumulate activation, emit terminal reward on timeout."""
        err_lin = self._lin_vel_error()
        err_ang = self._ang_vel_error()
        # Mean of two tanh activations — same shape as multi-task composer's
        # ``tracking_mean`` for the velocity preset (two tracking subtasks).
        act_lin = 1.0 - torch.tanh(err_lin / self.cfg.activation_std)
        act_ang = 1.0 - torch.tanh(err_ang / self.cfg.activation_std)
        per_step_activation = 0.5 * (act_lin + act_ang)

        self._sum_activation.add_(per_step_activation)
        self._transit_steps.add_(1)

        # Latch terminal reward one step before the outer ``time_out`` DoneTerm
        # fires — matches the ``is_timeout = buf >= effective_max - 1``
        # convention used by ``MultiTaskCommand`` so the timing is
        # apples-to-apples even under the adaptive episode-length curriculum.
        is_timeout = self._env.episode_length_buf >= self._effective_max_episode_length - 1
        transit_mean = self._sum_activation / self._transit_steps.clamp(min=1).to(self._sum_activation.dtype)
        self._task_reward = torch.where(is_timeout, transit_mean, torch.zeros_like(transit_mean))
        # ``task_done`` stays False — episodes end on timeout / base_contact,
        # not on command-success (no instant subtask to achieve).
        self._task_done.zero_()

        # Obs mirrors. Show world-frame xy lin-vel error and z ang-vel error so the
        # 3-vec ``command_track`` shape stays stable, even though the reward
        # uses the full 3D world-frame norm (which includes the z-bouncing
        # component the obs slot can't carry).
        cur_lin_w = wp.to_torch(self._robot.data.root_lin_vel_w)
        cur_ang_w = wp.to_torch(self._robot.data.root_ang_vel_w)
        self._command_track[:, :2] = self._target[:, :2] - cur_lin_w[:, :2]
        self._command_track[:, 2] = self._target[:, 2] - cur_ang_w[:, 2]
        self._progress.copy_(per_step_activation)

    # ------------------------------------------------------------------------
    # Helpers.
    # ------------------------------------------------------------------------

    def _lin_vel_error(self) -> torch.Tensor:
        # Mirror :class:`MultiTaskCommand`'s LIN_VEL_TRACKING kernel exactly:
        # 3D L2 norm of world-frame body linear velocity vs target ``[tx, ty, 0]``.
        # Using ``root_lin_vel_w`` (root body == "base" for all our legged URDFs)
        # is equivalent to the kernel's ``body_lin_vel_w[:, base_idx]``.
        cur = wp.to_torch(self._robot.data.root_lin_vel_w)  # [num_envs, 3]
        tgt = torch.zeros_like(cur)
        tgt[:, :2] = self._target[:, :2]
        return torch.linalg.vector_norm(cur - tgt, dim=-1)

    def _ang_vel_error(self) -> torch.Tensor:
        # Mirror :class:`MultiTaskCommand`'s ANG_VEL_TRACKING kernel exactly:
        # 3D L2 norm of world-frame body angular velocity vs target ``[0, 0, tz]``.
        cur = wp.to_torch(self._robot.data.root_ang_vel_w)  # [num_envs, 3]
        tgt = torch.zeros_like(cur)
        tgt[:, 2] = self._target[:, 2]
        return torch.linalg.vector_norm(cur - tgt, dim=-1)


@configclass
class MinimalVelocityCommandCfg(CommandTermCfg):
    """Cfg for :class:`MinimalVelocityCommand` — a debug control for the composer."""

    class_type: type = MinimalVelocityCommand
    asset_cfg: SceneEntityCfg = MISSING  # type: ignore[assignment]
    lin_vel_range: tuple[float, float] = (-1.0, 1.0)
    """Uniform xy lin-vel target range [m/s]. Matches ``LIN_VEL_TRACKING`` sampler."""
    ang_vel_range: tuple[float, float] = (-1.5, 1.5)
    """Uniform z ang-vel target range [rad/s]. Matches ``ANG_VEL_TRACKING`` sampler."""
    activation_std: float = 1.0
    """``A = 1 - tanh(err / activation_std)``. Match the multi-task preset for apples-to-apples."""
    tracking_episode_length_min_seconds: float | None = None
    """Adaptive episode-length curriculum lower bound [s].

    Mirrors :attr:`MultiTaskCfg.tracking_episode_length_min_seconds`. When
    ``None`` (default), every episode runs to ``env.max_episode_length``. When
    set, every reset draws a fresh per-env episode cap uniformly in
    ``[round(value / step_dt), max_episode_length]`` — same distribution as
    the multi-task composer's pure-tracking branch, so the running mean of
    ``error/body_lin_vel`` is comparable across the two implementations.
    """
