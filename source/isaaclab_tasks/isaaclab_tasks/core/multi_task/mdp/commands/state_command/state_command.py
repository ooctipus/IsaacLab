# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic state command lifecycle.

The command owns selected table rows plus fixed policy-command and error
buffers. The payload binds those opaque rows, performs any simulator reset
writes, and owns every domain-specific frame, target, success, and visualization
rule. Curriculum statistics belong to the curriculum term.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from tensordict import TensorDict

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

        self.randomize_command_indices: bool = cfg.randomize_command_indices

        # per-env selected table row; the curriculum may bind and overwrite it
        self.cmd_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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
    def states_relative(self) -> bool:
        """Deprecated mirror of :attr:`StateCommandCfg.states_relative`."""
        warnings.warn(
            "StateCommand.states_relative is deprecated; read StateCommand.cfg.states_relative.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.cfg.states_relative

    @property
    def command(self) -> torch.Tensor:
        """Policy-facing command observation written by the active payload."""
        return self._command

    @property
    def error(self) -> torch.Tensor:
        """Per-env, per-group command error written by the active payload."""
        return self._err

    @property
    def success_rates(self) -> torch.Tensor:
        """Return curriculum-owned success rates through the deprecated command boundary."""
        warnings.warn(
            "StateCommand.success_rates is deprecated; read the owning curriculum term's success_rates.",
            DeprecationWarning,
            stacklevel=2,
        )
        manager = getattr(self._env, "curriculum_manager", None)
        if manager is not None:
            for name in manager.active_terms:
                term = manager.get_term(name)
                if getattr(term, "sample_indices", None) is self.cmd_indices:
                    rates = getattr(term, "success_rates", None)
                    if isinstance(rates, torch.Tensor):
                        return rates
        raise RuntimeError("No curriculum term owns this StateCommand's selected rows.")

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

    def bind_rows(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        """Select exact table rows and run the normal payload binding lifecycle."""
        self.cmd_indices[env_ids] = task_rows
        self._payload.bind(env_ids, task_rows)
        self._payload.update(0.0, self._command, self._err)

    def get_target_obs_cache(self) -> TensorDict:
        """Return the new curriculum-owned target cache through a deprecated boundary."""
        warnings.warn(
            "StateCommand.get_target_obs_cache() is deprecated; bind the goal observation cache explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        manager = getattr(self._env, "curriculum_manager", None)
        if manager is None:
            raise RuntimeError("The environment has no curriculum manager with a goal observation cache.")
        try:
            return manager.get_term("goal_observations").observations
        except KeyError:
            raise RuntimeError(
                "No goal observation cache is configured; select the successor preset and bind its curriculum term."
            ) from None

    def get_spawn_obs_cache(self) -> TensorDict:
        """Return the new curriculum-owned spawn cache through a deprecated boundary."""
        warnings.warn(
            "StateCommand.get_spawn_obs_cache() is deprecated; bind the ValueShift observation cache explicitly.",
            DeprecationWarning,
            stacklevel=2,
        )
        manager = getattr(self._env, "curriculum_manager", None)
        if manager is not None:
            for name in manager.active_terms:
                term = manager.get_term(name)
                if getattr(term, "sample_indices", None) is self.cmd_indices:
                    try:
                        return term.value_shift.observation_cache
                    except RuntimeError:
                        break
        raise RuntimeError("No ValueShift observation cache owns this StateCommand's selected rows.")

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.long()
        if self.randomize_command_indices:
            task_rows = self.table.sample_rows(env_ids.numel())
        else:
            task_rows = self.cmd_indices[env_ids]
        self.bind_rows(env_ids, task_rows)

    def _update_command(self) -> None:
        self._payload.update(self._env.step_dt, self._command, self._err)

    def _update_metrics(self) -> None:
        for group_idx, name in enumerate(self._payload.error_names):
            self.metrics[name] = self._err[:, group_idx]

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        self._payload.set_debug_vis(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        self._payload.debug_visualize(self._env)
