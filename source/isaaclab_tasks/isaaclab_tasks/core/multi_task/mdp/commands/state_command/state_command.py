# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic state command: track selected task rows, delegate semantics.

The command owns only per-env lifecycle (the selected table row per env, the
policy-facing command observation, the error tensor, and the curriculum
success-rate buffer). Everything domain-specific lives behind two collaborators:

- a **task table** (built from ``cfg.task_table``) exposing ``num_tasks`` and
  ``gather(task_rows) -> (spawn_states, target_states)``;
- a **payload** (built from ``cfg.payload``) that owns its own buffers and
  implements target writing, observation/error computation, success/hold
  semantics, threshold reporting, metric aggregation, and debug drawing.

Factory assembly and legged locomotion supply different tables and payloads but
share this shell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from isaaclab.managers import CommandTerm

from isaaclab_tasks.core.multi_task.curriculum import set_reset_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .state_command_cfg import StateCommandCfg


class StateCommand(CommandTerm):
    """Track the selected task row per env and delegate command-state semantics."""

    cfg: StateCommandCfg

    def __init__(self, cfg: StateCommandCfg, env: ManagerBasedEnv):
        # table first, then the payload bound to that table (one construction
        # order for every domain). The factory table reads ``cfg.payload`` for
        # its reset-asset set, so it needs no constructed payload. Both are built
        # BEFORE ``super().__init__`` because the base initializer invokes
        # ``set_debug_vis`` -> ``_set_debug_vis_impl``, which delegates to the
        # payload. The builders take ``env``/``cfg`` directly, so no ``self``
        # state from the base is needed yet.
        self.table = cfg.task_table.class_type(cfg, env)
        self._payload = cfg.payload.class_type(cfg, env, self.table)

        super().__init__(cfg, env)

        self.states_relative: bool = cfg.states_relative
        self.randomize_command_indices: bool = cfg.randomize_command_indices

        # per-env selected table row; the curriculum may bind and overwrite it
        self.cmd_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # per-task curriculum success rate (written by the curriculum's monitor)
        self.success_rates = torch.zeros(self.table.num_tasks, device=self.device, dtype=torch.float32)
        # lazily-built [num_tasks, ...] caches of raw observations at each task's TARGET config (the goal
        # library for the z-conditioned successor critic: z = B(goal)) and SPAWN config (episode-start obs,
        # used e.g. by value-shift sampling). See :meth:`get_target_obs_cache` / :meth:`get_spawn_obs_cache`.
        self._target_obs_cache: TensorDict | None = None
        self._spawn_obs_cache: TensorDict | None = None

        self._command = torch.zeros(self.num_envs, self._payload.command_dim, device=self.device)
        self._err = torch.empty(self.num_envs, self._payload.error_dim, device=self.device)
        for name in self._payload.error_names:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

        self._resample_command(torch.arange(self.num_envs, device=self.device))
        self._update_command()

    def __str__(self) -> str:
        msg = f"{type(self).__name__}:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def payload(self):
        """The active domain payload (its public methods carry domain semantics)."""
        return self._payload

    @property
    def command(self) -> torch.Tensor:
        """Policy-facing command observation written by the active payload."""
        return self._command

    @property
    def error(self) -> torch.Tensor:
        """Per-env, per-group command error written by the active payload."""
        return self._err

    @property
    def command_std(self) -> torch.Tensor:
        """Per-env success thresholds for the currently-bound rows ``[N, error_dim]``."""
        return self._payload.command_std()

    def get_task_done(self) -> torch.Tensor:
        """Per-env task-success flag (payload-defined)."""
        return self._payload.get_task_done()

    def get_task_reward(self) -> torch.Tensor:
        """Per-env task-success reward (payload-defined)."""
        return self._payload.get_task_reward()

    @torch.no_grad()
    def get_target_obs_cache(self) -> TensorDict:
        """Raw observations at every task's TARGET config (delta-0 goal), ``[num_tasks, ...]``, built once.

        The z-conditioned successor critic conditions on the goal via ``z = B(goal)``, where the goal is the
        observation AT a task's target config (target proprioception + perception). Cached after the first call
        (the table is fixed); the learner recomputes ``B(cache)`` live each update (the encoder keeps training,
        so raw obs -- not embeddings -- are cached).
        """
        if self._target_obs_cache is None:
            self._target_obs_cache = self._build_obs_cache(use_target=True)
        return self._target_obs_cache

    @torch.no_grad()
    def get_spawn_obs_cache(self) -> TensorDict:
        """Raw observations at every task's SPAWN config, ``[num_tasks, ...]``, built once.

        The observation a fresh episode of each task starts from -- e.g. the value-shift sampling strategy
        scores per-task value drift at this episode-start observation. Cached after the first call.
        """
        if self._spawn_obs_cache is None:
            self._spawn_obs_cache = self._build_obs_cache(use_target=False)
        return self._spawn_obs_cache

    @torch.no_grad()
    def _build_obs_cache(self, use_target: bool) -> TensorDict:
        """Sweep all :attr:`table.num_tasks` tasks and cache the observation at each task's SPAWN or TARGET
        state, ``[num_tasks, ...]``.

        Each env-sized batch teleports the reset assets to the chosen state (reusing :func:`set_reset_state`,
        the same write the per-env reset uses), settles the sim, and force-recomputes ray-cast sensors --
        ``scene.update(dt=0.0)`` does NOT advance their lazy update timer, so the height scan would otherwise be
        stale at the teleported pose. Live env state is saved and restored so the sweep leaves running episodes
        untouched.
        """
        env = self._env
        num_envs, num_tasks, device = self.num_envs, self.table.num_tasks, self.device
        reset_assets = self._payload.reset_assets

        # Save live state of every reset asset (+ the selected rows) so the sweep is non-destructive.
        saved = {}
        for name in reset_assets:
            asset = env.scene[name]
            jpos = getattr(asset.data, "joint_pos", None)
            saved[name] = (
                asset.data.root_state_w.clone(),
                None if jpos is None else jpos.clone(),
                None if jpos is None else asset.data.joint_vel.clone(),
            )
        saved_cmd = self.cmd_indices.clone()

        cache: TensorDict | None = None
        all_env_ids = torch.arange(num_envs, device=device)
        for start in range(0, num_tasks, num_envs):
            task_ids = torch.arange(start, min(start + num_envs, num_tasks), device=device)
            env_ids = all_env_ids[: task_ids.numel()]
            spawn_states, target_states = self.table.gather(task_ids)
            states = target_states if use_target else spawn_states
            set_reset_state(env, states, env_ids, reset_assets, is_relative=self.states_relative)
            env.sim.forward()
            env.scene.update(dt=0.0)
            for sensor in env.scene.sensors.values():
                sensor.update(dt=0.0, force_recompute=True)
            obs = env.observation_manager.compute()
            if cache is None:
                cache = TensorDict(
                    {g: torch.zeros((num_tasks, *t.shape[1:]), dtype=t.dtype, device=device) for g, t in obs.items()},
                    batch_size=[num_tasks],
                )
            for g, t in obs.items():
                cache[g][task_ids] = t[env_ids]

        # Restore live state.
        for name, (root, jpos, jvel) in saved.items():
            asset = env.scene[name]
            asset.write_root_state_to_sim(root)
            if jpos is not None:
                asset.write_joint_state_to_sim(jpos, jvel)
        self.cmd_indices.copy_(saved_cmd)
        env.sim.forward()
        env.scene.update(dt=0.0)
        return cache

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.long()
        if self.randomize_command_indices:
            self.cmd_indices[env_ids] = torch.randint(0, self.table.num_tasks, (env_ids.numel(),), device=self.device)

        task_rows = self.cmd_indices[env_ids]
        spawn_states, target_states = self.table.gather(task_rows)
        # explicit frame: env-local states are lifted to world by ``env_origins``
        # (per-asset for the spawn write, per-payload for the target)
        target_origin = self._env.scene.env_origins[env_ids] if self.states_relative else None
        self._payload.resample(env_ids, task_rows, target_states, target_origin)
        set_reset_state(self._env, spawn_states, env_ids, self._payload.reset_assets, is_relative=self.states_relative)
        self._payload.update(0.0, self._command, self._err)

    def _update_command(self) -> None:
        self._payload.update(self._env.step_dt, self._command, self._err)

    def _update_metrics(self) -> None:
        for group_idx, name in enumerate(self._payload.error_names):
            self.metrics[name] = self._err[:, group_idx]
        self._payload.log_metrics(self._env, self.success_rates)

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        self._payload.set_debug_vis(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        self._payload.debug_visualize(self._env)
