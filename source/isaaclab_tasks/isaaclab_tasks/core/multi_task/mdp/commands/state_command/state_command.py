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

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .state_command_cfg import StateCommandCfg


class StateCommand(CommandTerm):
    """Track the selected task row per env and delegate command-state semantics."""

    cfg: StateCommandCfg

    def __init__(self, cfg: StateCommandCfg, env: ManagerBasedRLEnv):
        # Table first, then the payload bound to that table (one construction
        # order for every domain). Both are built BEFORE ``super().__init__``
        # because the base initializer invokes
        # ``set_debug_vis`` -> ``_set_debug_vis_impl``, which delegates to the
        # payload. Table construction receives resolved configuration and a
        # device, never a live scene.
        self.table = cfg.task_table.build(cfg, env.cfg.scene, env.device)
        self._payload = cfg.payload.class_type(cfg, env, self.table)

        super().__init__(cfg, env)

        self.randomize_command_indices: bool = cfg.randomize_command_indices

        # per-env selected table row; the curriculum may bind and overwrite it
        self.cmd_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._command = torch.zeros(self.num_envs, self._payload.command_dim, device=self.device)
        self._err = torch.empty(self.num_envs, self._payload.error_dim, device=self.device)
        self._update_step = env.common_step_counter
        for group_idx, name in enumerate(self._payload.error_names):
            self.metrics[name] = self._err[:, group_idx]

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
        self._refresh()
        return self._command

    @property
    def error(self) -> torch.Tensor:
        """Per-env, per-group command error written by the active payload."""
        self._refresh()
        return self._err

    @property
    def command_std(self) -> torch.Tensor:
        """Return success thresholds when the active payload defines them."""
        command_std = getattr(self._payload, "command_std", None)
        if command_std is None:
            raise NotImplementedError(f"{type(self._payload).__name__} does not define success thresholds.")
        return command_std()

    def get_task_done(self) -> torch.Tensor:
        """Return task success when the active payload defines it."""
        get_task_done = getattr(self._payload, "get_task_done", None)
        if get_task_done is None:
            raise NotImplementedError(f"{type(self._payload).__name__} does not define task success.")
        self._refresh()
        return get_task_done()

    def get_task_reward(self) -> torch.Tensor:
        """Return task reward when the active payload defines it."""
        get_task_reward = getattr(self._payload, "get_task_reward", None)
        if get_task_reward is None:
            raise NotImplementedError(f"{type(self._payload).__name__} does not define task reward.")
        self._refresh()
        return get_task_reward()

    def get_state(self, name: str) -> torch.Tensor:
        """Return named domain state when the active payload defines it."""
        get_state = getattr(self._payload, "get_state", None)
        if get_state is None:
            raise NotImplementedError(f"{type(self._payload).__name__} does not define named command state.")
        self._refresh()
        return get_state(name)

    def bind_rows(
        self,
        env_ids: torch.Tensor,
        task_rows: torch.Tensor,
    ) -> None:
        """Bind exact table rows through one command-owned transaction.

        Args:
            env_ids: Environment rows receiving new tasks.
            task_rows: Task-table rows paired with :paramref:`env_ids`.
        """
        self._refresh()
        self.cmd_indices[env_ids] = task_rows
        self._payload.bind(env_ids, task_rows)
        self.materialize()

    def bind_rows_target(self, env_ids: torch.Tensor, task_rows: torch.Tensor) -> None:
        """Bind target physics and command state for cold observation materialization.

        Args:
            env_ids: Environment rows receiving target states.
            task_rows: Task-table rows paired with :paramref:`env_ids`.
        """
        self._refresh()
        self.cmd_indices[env_ids] = task_rows
        self._payload.bind_target(env_ids, task_rows)
        self.materialize()

    def materialize(self) -> None:
        """Refresh command outputs after a cold simulator-state materialization."""
        self._payload.update(0.0, self._command, self._err)
        self._update_step = self._env.common_step_counter

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.long()
        if self.randomize_command_indices:
            task_rows = self._payload.sample_rows(env_ids.numel())
        else:
            task_rows = self.cmd_indices[env_ids]
        self.bind_rows(env_ids, task_rows)

    def _refresh(self) -> None:
        """Update payload state once for the latest completed control step."""
        step = self._env.common_step_counter
        if self._update_step == step:
            return
        self._payload.update(self._env.step_dt, self._command, self._err)
        self._update_step = step

    def _update_command(self) -> None:
        self._refresh()

    def _update_metrics(self) -> None:
        self._refresh()

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        self._payload.set_debug_vis(debug_vis)

    def _debug_vis_callback(self, event) -> None:
        self._payload.debug_visualize(self._env)
