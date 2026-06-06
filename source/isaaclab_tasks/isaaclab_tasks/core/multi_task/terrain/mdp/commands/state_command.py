# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Track selected task rows and delegate command-state semantics to a payload."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm

from isaaclab_tasks.core.multi_task.curriculum import set_reset_state

from .task_table_builder import RelativeStateTaskTable

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import RelativeStateCommandCfg


class RelativeStateCommand(CommandTerm):
    """Track the delta between a target and current robot state.

    The command owns lifecycle tensors. The configured payload owns command
    semantics: target writing, observation writing, error grouping, success,
    and debug drawing.
    """

    cfg: RelativeStateCommandCfg
    TaskTable = RelativeStateTaskTable

    def __init__(self, cfg: RelativeStateCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.table = cfg.task_table.class_type(cfg, env)
        self._payload = cfg.payload.class_type(cfg, env, self.table)

        self._command_names = list(cfg.commands.keys())
        self.success_rates: torch.Tensor = torch.zeros(self.table.num_tasks, device=self.device, dtype=torch.float32)
        self._success_per_cmd = torch.zeros(self.table.kind.shape[0], device=self.device)
        self.cmd_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # task table row per env
        self.randomize_command_indices: bool = cfg.randomize_command_indices

        self.cmd_buf = torch.zeros(self.num_envs, 3, self._payload.state_dim, device=self.device).contiguous()
        self.cmd_buf[:, 1] = 1.0
        self.cmd_mask = torch.zeros(self.num_envs, self._payload.mask_dim, device=self.device, dtype=torch.bool)
        self._command = torch.zeros(self.num_envs, self._payload.command_dim, device=self.device)
        self._err = torch.empty(self.num_envs, self._payload.error_dim, device=self.device)

        for name in self._payload.error_names:
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "RelativeStateCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Policy-facing command observation written by the active payload."""
        return self._command

    @property
    def target_state(self) -> torch.Tensor:
        """Payload target state rows."""
        return self.cmd_buf[:, 0]

    @property
    def cmd_ids(self) -> torch.Tensor:
        """Command-type ids derived from the currently selected task rows."""
        return torch.bucketize(self.cmd_indices, self.table.offsets[1:-1], right=True)

    @property
    def command_std(self) -> torch.Tensor:
        """Per-env success thresholds for the currently-bound task.

        The shape and group order are defined by the active payload. Values
        use the same per-task units as the success criteria.
        """
        return self._payload.command_std(self.cmd_ids)

    def current_state_env(self, env_origins: torch.Tensor) -> torch.Tensor:
        """Current payload state in the per-env local frame."""
        return self._payload.current_state_env(self.cmd_buf[:, 2], env_origins)

    def target_state_env(self, env_origins: torch.Tensor) -> torch.Tensor:
        """Target payload state in the per-env local frame."""
        return self._payload.target_state_env(self.cmd_buf[:, 0], env_origins)

    def _update_metrics(self):
        error = self._err
        for group_idx, name in enumerate(self._payload.error_names):
            self.metrics[name] = error[:, group_idx]

        log = self._env.extras.setdefault("log", {})
        for cmd_id, name in enumerate(self._command_names):
            start = int(self.table.offsets[cmd_id].item())
            end = int(self.table.offsets[cmd_id + 1].item())
            if end > start:
                self._success_per_cmd[cmd_id] = self.success_rates[start:end].mean()
            else:
                self._success_per_cmd[cmd_id] = 0.0
            log["Metrics/goal_point/success_rate_" + name] = self._success_per_cmd[cmd_id].item()

    def _resample_command(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return

        env_ids = env_ids.long()
        num_resets = env_ids.numel()
        if self.randomize_command_indices:
            self.cmd_indices[env_ids] = torch.randint(0, self.table.num_tasks, (num_resets,), device=self.device)

        task_rows = self.cmd_indices[env_ids]
        target_states = self.table.spawn_states[self.table.target_index[task_rows]].clone()
        spawn_states = self.table.spawn_states[self.table.spawn_index[task_rows]].clone()
        origin_offsets = self._task_state_origin_offsets(env_ids)
        if origin_offsets is not None:
            target_states[:, :3].add_(origin_offsets)
            spawn_states[:, :3].add_(origin_offsets)

        self.cmd_mask.index_copy_(0, env_ids, self.table.task_mask[task_rows])
        self._payload.resample(env_ids, task_rows, target_states, self.cmd_buf)
        set_reset_state(self._env, spawn_states, env_ids, self._payload.reset_assets)
        self._payload.update(self.cmd_ids, 0.0, self.cmd_buf, self.cmd_mask, self._command, self._err)

    def _task_state_origin_offsets(self, env_ids: torch.Tensor) -> torch.Tensor | None:
        """Return the env-replica offset for task-table states, if any."""
        terrain = self._env.scene.terrain
        terrain_prim_path = getattr(getattr(terrain, "cfg", None), "prim_path", "")
        env_regex_ns = getattr(self._env.scene, "env_regex_ns", "")
        env_ns = getattr(self._env.scene, "env_ns", "")
        is_replicated_terrain = (env_regex_ns and env_regex_ns in terrain_prim_path) or (
            env_ns and terrain_prim_path.startswith(f"{env_ns}/")
        )
        if not is_replicated_terrain:
            return None
        default_env_origins = getattr(self._env.scene, "_default_env_origins", None)
        if default_env_origins is not None:
            return default_env_origins[env_ids]
        return self._env.scene.env_origins[env_ids]

    def _update_command(self):
        """Recompute delta state from current robot state. Minimizes temporaries."""
        self._payload.update(self.cmd_ids, self._env.step_dt, self.cmd_buf, self.cmd_mask, self._command, self._err)

    def get_task_done(self) -> torch.Tensor:
        return self.cmd_buf[:, 1, self._payload.time_idx] <= 0.0

    def get_task_reward(self) -> torch.Tensor:
        return (self.cmd_buf[:, 1, self._payload.time_idx] <= 0.0).float()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            from isaaclab.markers import VisualizationMarkers

            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            if not hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
            if hasattr(self, "current_vel_visualizer"):
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self, "table"):
            return
        self._payload.debug_visualize(
            env=self._env,
            cmd_ids=self.cmd_ids,
            cmd_buf=self.cmd_buf,
            goal_visualizer=self.goal_visualizer,
            current_vel_visualizer=self.current_vel_visualizer,
        )
