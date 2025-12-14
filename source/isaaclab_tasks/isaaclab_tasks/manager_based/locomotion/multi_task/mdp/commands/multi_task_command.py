from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm

from .kernels import ACTIVATION_KERNELS, METRIC_KERNELS

if TYPE_CHECKING:
    from .commands_cfg import MultiTaskCfg


def pad_index_rows(index_rows: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad ragged lists of indices into a dense [num_rows, max_len] table.

    Returns:
        index_table: int64 tensor with -1 padding.
        valid_table: bool mask indicating which entries in each row are valid.
    """
    max_len = max((len(row) for row in index_rows), default=0)
    if max_len == 0:
        index_table = torch.full((len(index_rows), 1), -1, dtype=torch.long, device=device)
        valid_table = torch.zeros((len(index_rows), 1), dtype=torch.bool, device=device)
        return index_table, valid_table

    index_table = torch.full((len(index_rows), max_len), -1, dtype=torch.long, device=device)
    valid_table = torch.zeros((len(index_rows), max_len), dtype=torch.bool, device=device)

    for row_index, row in enumerate(index_rows):
        if not row:
            continue
        count = len(row)
        index_table[row_index, :count] = torch.tensor(row, dtype=torch.long, device=device)
        valid_table[row_index, :count] = True

    return index_table, valid_table


@dataclass
class TaskSpec:
    """Compiled description of all tasks and shared subtasks.

    - Each task has a padded list of subtask IDs.
    - Each subtask stores which asset it refers to and which kernels to use.
    """

    # tasks -> subtasks (padded)
    task_names: list[str]
    task_subtask_ids: torch.Tensor  # [num_tasks, max_subtasks_per_task], -1 padded
    task_subtask_valid: torch.Tensor  # [num_tasks, max_subtasks_per_task]

    # global subtasks
    asset_names: list[str]  # [num_assets]
    subtask_asset_id: torch.Tensor  # [num_subtasks]
    state_kernel_names: list[str]  # [num_subtasks]
    state_kernel_id: torch.Tensor  # [num_subtasks]
    metric_kernel_names: list[str]  # [num_subtasks]
    metric_kernel_id: torch.Tensor  # [num_subtasks]
    activation_kernel_names: list[str]  # [num_subtasks]
    activation_kernel_id: torch.Tensor  # [num_subtasks]
    activation_kernel_param: torch.Tensor  # [num_subtasks]
    is_tracking: torch.Tensor  # [num_subtasks]
    is_instant: torch.Tensor  # [num_subtasks]


class MultiTaskCommand(CommandTerm):
    """Multi-task command / reward engine.

    Compiles per-task reward subtasks into a shared [env, subtask] table and
    evaluates tracking / reach / mixed rewards in a fully batched way.
    """

    def __init__(self, cfg: MultiTaskCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.gamma = 0.99
        self.max_episode_length = float(env.max_episode_length)

        # Build static subtask / task layout
        self.spec = self._build_spec()
        self.num_tasks = len(self.spec.task_names)
        self.num_subtasks = int(self.spec.activation_kernel_id.numel())

        # Per-env state
        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self.task_samples = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)

        # Pre-allocated work buffers for reward computation
        max_subtasks = self.spec.task_subtask_ids.shape[1]
        max_envs = self.num_envs  # could be an upper bound instead

        self._buf_safe_ids = torch.empty((max_envs, max_subtasks), dtype=torch.long, device=self.device)
        self._buf_selected_value = torch.empty((max_envs, max_subtasks), dtype=torch.float32, device=self.device)
        self._buf_is_tracking = torch.empty((max_envs, max_subtasks), dtype=torch.bool, device=self.device)
        self._buf_is_instant = torch.empty((max_envs, max_subtasks), dtype=torch.bool, device=self.device)
        self._buf_instant_factor = torch.empty((max_envs, max_subtasks), dtype=torch.float32, device=self.device)
        self._buf_reward = torch.empty((max_envs,), dtype=torch.float32, device=self.device)

    def _subtask_signature(self, subtask_cfg: MultiTaskCfg.BaseTaskCfg) -> tuple:
        """Return a hashable key that uniquely identifies a subtask type.

        Used to deduplicate subtasks that are semantically identical across tasks.
        """
        return (
            type(subtask_cfg).__name__,
            subtask_cfg.asset_cfg.name,
            int(subtask_cfg.state_kernel),
            int(subtask_cfg.metric_kernel),
            int(subtask_cfg.activation_kernel),
            float(subtask_cfg.activation_kernel_param),
        )

    def _build_spec(self) -> TaskSpec:
        """Compile task/subtask configs into a dense TaskSpec on the current device."""
        from .commands_cfg import MultiTaskCfg

        cfg: MultiTaskCfg = self.cfg
        device = self.device

        task_names = list(cfg.tasks.keys())
        signature_to_subtask_id: dict[tuple, int] = {}

        state_kernel_names: list[str] = []
        metric_kernel_names: list[str] = [k.__name__ for k in METRIC_KERNELS]
        activation_kernel_names: list[str] = [k.__name__ for k in ACTIVATION_KERNELS]

        subtask_asset_name: list[str] = []
        state_kernel_id: list[int] = []
        metric_kernel_id: list[int] = []
        activation_kernel_id: list[int] = []
        activation_kernel_param: list[float] = []
        is_tracking: list[bool] = []
        is_instant: list[bool] = []

        task_to_subtask_ids: list[list[int]] = []

        for task_name in task_names:
            subtask_ids_for_task: list[int] = []
            for subtask_cfg in cfg.tasks[task_name]:
                signature = self._subtask_signature(subtask_cfg)
                subtask_id = signature_to_subtask_id.get(signature)

                # Create new global subtask if we haven't seen this signature yet
                if subtask_id is None:
                    subtask_id = len(state_kernel_id)
                    signature_to_subtask_id[signature] = subtask_id

                    subtask_asset_name.append(subtask_cfg.asset_cfg.name)
                    state_kernel_id.append(int(subtask_cfg.state_kernel))
                    metric_kernel_id.append(int(subtask_cfg.metric_kernel))
                    activation_kernel_id.append(int(subtask_cfg.activation_kernel))
                    activation_kernel_param.append(float(subtask_cfg.activation_kernel_param))
                    is_tracking.append(isinstance(subtask_cfg, MultiTaskCfg.TrackingTaskCfg))
                    is_instant.append(isinstance(subtask_cfg, MultiTaskCfg.InstantaneousTaskCfg))

                subtask_ids_for_task.append(subtask_id)

            if not subtask_ids_for_task:
                raise ValueError(f"Task '{task_name}' has no subtasks.")
            task_to_subtask_ids.append(subtask_ids_for_task)

        task_subtask_ids, task_subtask_valid = pad_index_rows(task_to_subtask_ids, device=device)

        # Map asset names to contiguous IDs
        asset_names: list[str] = []
        asset_name_to_id: dict[str, int] = {}
        subtask_asset_id_list: list[int] = []

        for name in subtask_asset_name:
            asset_id = asset_name_to_id.get(name)
            if asset_id is None:
                asset_id = len(asset_names)
                asset_names.append(name)
                asset_name_to_id[name] = asset_id
            subtask_asset_id_list.append(asset_id)

        return TaskSpec(
            task_names=task_names,
            task_subtask_ids=task_subtask_ids,
            task_subtask_valid=task_subtask_valid,
            asset_names=asset_names,
            subtask_asset_id=torch.tensor(subtask_asset_id_list, dtype=torch.long, device=device),
            state_kernel_names=state_kernel_names,
            state_kernel_id=torch.tensor(state_kernel_id, dtype=torch.long, device=device),
            metric_kernel_names=metric_kernel_names,
            metric_kernel_id=torch.tensor(metric_kernel_id, dtype=torch.long, device=device),
            activation_kernel_names=activation_kernel_names,
            activation_kernel_id=torch.tensor(activation_kernel_id, dtype=torch.long, device=device),
            activation_kernel_param=torch.tensor(activation_kernel_param, dtype=torch.float32, device=device),
            is_tracking=torch.tensor(is_tracking, dtype=torch.bool, device=device),
            is_instant=torch.tensor(is_instant, dtype=torch.bool, device=device),
        )

    @property
    def command(self) -> torch.Tensor:
        """Current command tensor (placeholder for now)."""
        return self._command

    def resample_indices(self, env_ids: torch.Tensor) -> None:
        """Sample a new task index for the given envs."""
        self.task_samples[env_ids] = torch.randint(0, self.num_tasks, (env_ids.numel(),), device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Hook for sampling commands per env (not implemented yet)."""
        pass

    def _update_command(self):
        """Hook for post-processing commands (not implemented yet)."""
        pass

    def _update_metrics(self):
        """Hook for updating metrics per step (not implemented yet)."""
        pass

    def _dummy_error(self) -> torch.Tensor:
        """
        Deterministic dummy 'metric error' just to validate logic:
        error[env, subtask] in [0, 2).
        """
        env_index = torch.arange(self.num_envs, device=self.device, dtype=torch.float32).unsqueeze(1)
        subtask_index = torch.arange(self.num_subtasks, device=self.device, dtype=torch.float32).unsqueeze(0)
        return (0.01 * env_index + 0.03 * subtask_index) % 2.0

    def _apply_activation_kernels(self, error: torch.Tensor) -> torch.Tensor:
        """Apply per-subtask activation kernels to error -> [0, 1] 'goodness' scores."""
        activated = torch.empty_like(error)
        for kernel_index, kernel_fn in enumerate(ACTIVATION_KERNELS):
            column_mask = self.spec.activation_kernel_id == kernel_index
            if not column_mask.any():
                continue
            param = self.spec.activation_kernel_param[column_mask]
            out = kernel_fn(error[:, column_mask], param)
            activated[:, column_mask] = out.to(error.dtype)  # bool -> float
        return activated

    def get_task_done(self) -> torch.Tensor:
        """Return done flags for each env (no task-driven termination yet)."""
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def get_task_reward(self) -> torch.Tensor:
        """Compute per-env multi-task reward from activated subtasks.

        Handles three cases:
            - tracking-only tasks,
            - instantaneous-only tasks (reach),
            - mixed tasks (tracking + reach).
        """
        spec = self.spec
        num_envs = self.num_envs
        max_subtasks = spec.task_subtask_ids.shape[1]

        # 1) error → per-subtask activation
        error = self._dummy_error()  # [env, subtask]
        activated = self._apply_activation_kernels(error)  # [env, subtask]

        # 2) gather values for this env's subtasks
        safe_ids, selected_valid = self._select_subtasks()

        selected_value = self._buf_selected_value[:num_envs, :max_subtasks]
        selected_value.copy_(activated.gather(1, safe_ids))
        selected_value.masked_fill_(~selected_valid, 0.0)

        selected_is_tracking = self._buf_is_tracking[:num_envs, :max_subtasks]
        selected_is_instant = self._buf_is_instant[:num_envs, :max_subtasks]

        selected_is_tracking.copy_(spec.is_tracking[safe_ids])
        selected_is_instant.copy_(spec.is_instant[safe_ids])

        selected_is_tracking &= selected_valid
        selected_is_instant &= selected_valid

        has_tracking = selected_is_tracking.any(dim=1)
        has_instant = selected_is_instant.any(dim=1)

        # 3) tracking-only term (Continue-style)
        tracking_mean = self._compute_tracking_mean(selected_value, selected_is_tracking)
        tracking_reward = ((1.0 - self.gamma) / self.max_episode_length) * tracking_mean

        # 4) instantaneous reach success (Reach-style)
        reach_success = self._compute_reach_success(selected_value, selected_is_instant)

        # 5) mixed case: tracking + reach bonus with early-success ramp
        episode_step = self._env.episode_length_buf.to(torch.float32)
        ramp = 1.0 - episode_step / self.max_episode_length
        mixed_reward = (1.0 / self.max_episode_length) * tracking_mean + reach_success * (
            1.0 + ramp * tracking_mean
        )

        # 6) pick the right recipe per env
        reward = self._buf_reward[:num_envs]
        reward.zero_()

        mask = has_tracking & ~has_instant
        reward[mask] = tracking_reward[mask]

        mask = ~has_tracking & has_instant
        reward[mask] = reach_success[mask]

        mask = has_tracking & has_instant
        reward[mask] = mixed_reward[mask]

        return reward

    def _select_subtasks(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve per-env padded subtask IDs and valid mask into the safe_ids buffer."""
        spec = self.spec
        num_envs = self.num_envs
        max_subtasks = spec.task_subtask_ids.shape[1]

        # [env, max_subtasks]
        safe_ids = self._buf_safe_ids[:num_envs, :max_subtasks]
        selected_subtask_ids = spec.task_subtask_ids[self.task_samples]
        safe_ids.copy_(selected_subtask_ids)
        safe_ids.clamp_(min=0)

        selected_valid = spec.task_subtask_valid[self.task_samples]
        return safe_ids, selected_valid

    @staticmethod
    def _compute_tracking_mean(values: torch.Tensor, is_tracking: torch.Tensor) -> torch.Tensor:
        """Mean activation over tracking subtasks per env."""
        tracking_count = is_tracking.sum(dim=1).clamp_min(1)
        return (values * is_tracking).sum(dim=1) / tracking_count

    def _compute_reach_success(self, values: torch.Tensor, is_instant: torch.Tensor) -> torch.Tensor:
        """AND-like success over instantaneous subtasks via product across subtasks."""
        num_envs, max_subtasks = values.shape
        instant_factor = self._buf_instant_factor[:num_envs, :max_subtasks]
        instant_factor.copy_(values)
        instant_factor[~is_instant] = 1.0
        return instant_factor.prod(dim=1)
