from __future__ import annotations
from dataclasses import dataclass

import inspect
import torch
import torch.nn.functional as F
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, SceneEntityCfg

from .success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command_cfg import AssemblyTaskCommandCfg
    from .data import AlignmentMetric


class AssemblyTaskCommand(CommandTerm):
    cfg: AssemblyTaskCommandCfg

    @dataclass
    class TaskType:
        Align: int = 0
        Insert: int = 1
        Screw: int = 2

    def __init__(self, cfg: AssemblyTaskCommandCfg, env: DataManagerBasedRLEnv):
        super().__init__(cfg, env)
        # extract the robot and body index for which the command is generated
        self._env: DataManagerBasedRLEnv = env
        self.tasks: list[str] = cfg.tasks
        self.task_categories = self._categorize_tasks(self.tasks)
        self.num_tasks = len(self.tasks)
        self.task_id = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)
        self.cur_task_categories = self.task_categories[self.task_id]
        self.last_task_id = self.task_id.clone()
        # -- metrics
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)
        self.reset_terms_when_resample = cfg.reset_terms_when_resample

        for name, term_cfg in self.reset_terms_when_resample.items():
            if term_cfg.mode != "reset":
                raise ValueError(f"Term '{name}' in 'reset_terms_when_resample' must have mode 'reset'.")
            if inspect.isclass(term_cfg.func):
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
            for value in term_cfg.params.values():
                if isinstance(value, SceneEntityCfg):
                    value.resolve(env.scene)

        success_monitor_cfg = SuccessMonitorCfg(
            monitored_history_len=100,
            num_monitored_data=self.num_tasks,
            device=env.device
        )
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)
        self.alignment_metric_cfg = cfg.alignment_metric_cfg
        self.success_recorder = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        for i in range(self.num_tasks):
            self.metrics[f"pos_error_{self.tasks[i]}"] = torch.zeros(env.num_envs, device=env.device)
            self.metrics[f"rot_error_{self.tasks[i]}"] = torch.zeros(env.num_envs, device=env.device)

    def __str__(self) -> str:
        msg = "AssemblyTaskCommand:\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    def _categorize_tasks(self, tasks: list[str]) -> torch.Tensor:
        # Categorize tasks based on their type
        task_types = torch.zeros(len(tasks), dtype=torch.int, device=self.device)
        for i, task in enumerate(tasks):
            if "Insert" in task or "GearMesh" in task:
                task_types[i] = self.TaskType.Insert
            elif "Align" in task:
                task_types[i] = self.TaskType.Align
            elif "Thread" in task:
                task_types[i] = self.TaskType.Screw
        return task_types

    @property
    def command(self):
        return self.task_id

    @property
    def one_hot_command(self) -> torch.Tensor:
        return F.one_hot(self.task_id, num_classes=self.num_tasks).float()

    def _update_metrics(self):
        # logs end of episode data
        reset_env = self._env.episode_length_buf == 0
        if torch.any(reset_env):
            success_mask = torch.where(self.success_recorder[reset_env], 1.0, 0.0)
            self.metrics["success"][reset_env] = self.success_recorder[reset_env].to(torch.float)
            reset_task_ids = self.last_task_id[reset_env]
            self.success_monitor.success_update(reset_task_ids, success_mask)
        alignment_data: AlignmentMetric.AlignmentData = self.alignment_metric_cfg.get(self._env.data_manager)
        task_success_rate = self.success_monitor.get_success_rate()
        log = {f"Metrics/task_success/{self.tasks[i]}": task_success_rate[i].item() for i in range(self.num_tasks)}
        normalized_pos_error = alignment_data.pos_error.norm(dim=1)
        normalized_rot_error = alignment_data.rot_error.norm(dim=1)
        self.success_recorder[:] = alignment_data.pos_aligned & alignment_data.rot_aligned
        for i in range(self.num_tasks):
            task_mask = self.task_id == i
            self.metrics[f"pos_error_{self.tasks[i]}"][task_mask] = normalized_pos_error[task_mask]
            self.metrics[f"rot_error_{self.tasks[i]}"][task_mask] = normalized_rot_error[task_mask]

        self._env.extras["log"].update(log)

    def _resample_command(self, env_ids: Sequence[int]):
        self.last_task_id[env_ids] = self.task_id[env_ids]
        sampled_indices = self.success_monitor.failure_rate_sampling(env_ids)
        self.task_id[env_ids] = sampled_indices.to(self.task_id.dtype)
        self.cur_task_categories[env_ids] = self.task_categories[sampled_indices]
        for name, term in self._env.data_manager._terms.items():
            term.command_reset(env_ids)

        for name, term in self.reset_terms_when_resample.items():
            func = term.func
            func(self._env, env_ids, **term.params)
            for name, term in self._env.data_manager._terms.items():
                term._update_data(env_ids)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
