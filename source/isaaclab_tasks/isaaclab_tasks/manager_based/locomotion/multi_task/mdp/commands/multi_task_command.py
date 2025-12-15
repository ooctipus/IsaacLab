from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import CommandTerm, SceneEntityCfg

from .kernels import ACTIVATION_KERNELS, METRIC_KERNELS, STATE_KERNELS, SAMPLER_KERNELS

if TYPE_CHECKING:
    from .commands_cfg import MultiTaskCfg


def pad_index_rows(index_rows: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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


def _ids_sig(ids) -> tuple:
    """Stable signature for ids (list/tensor/slice/None)."""
    if ids is None:
        return ()
    if isinstance(ids, slice):
        return ("ALL",)
    if torch.is_tensor(ids):
        return tuple(int(x) for x in ids.tolist())
    return tuple(int(x) for x in ids)


@dataclass
class TaskSpec:
    task_names: list[str]
    task_subtask_ids: torch.Tensor
    task_subtask_valid: torch.Tensor

    # per-subtask kernel selections
    state_kernel_id: torch.Tensor
    metric_kernel_id: torch.Tensor
    sampler_kernel_id: torch.Tensor
    sampler_kernel_param: torch.Tensor          # [S, Pmax] padded
    sampler_kernel_param_len: torch.Tensor      # [S] original unpadded length
    activation_kernel_id: torch.Tensor
    activation_kernel_param: torch.Tensor

    is_tracking: torch.Tensor
    is_instant: torch.Tensor

    # scene bindings
    subtask_asset_cfgs: list[SceneEntityCfg]
    subtask_entity_id: torch.Tensor             # [S], groups identical (asset_cfg.name + ids)


class MultiTaskCommand(CommandTerm):
    def __init__(self, cfg: MultiTaskCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.gamma = 0.99
        self.max_episode_length = float(env.max_episode_length)

        self.spec = self._build_spec()
        self.num_tasks = len(self.spec.task_names)
        self.num_subtasks = int(self.spec.state_kernel_id.numel())

        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self.task_samples = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)

        # work buffers
        max_subtasks = self.spec.task_subtask_ids.shape[1]
        self._buf_safe_ids = torch.empty((self.num_envs, max_subtasks), dtype=torch.long, device=self.device)
        self._buf_selected_value = torch.empty((self.num_envs, max_subtasks), dtype=torch.float32, device=self.device)
        self._buf_is_tracking = torch.empty((self.num_envs, max_subtasks), dtype=torch.bool, device=self.device)
        self._buf_is_instant = torch.empty((self.num_envs, max_subtasks), dtype=torch.bool, device=self.device)
        self._buf_instant_factor = torch.empty((self.num_envs, max_subtasks), dtype=torch.float32, device=self.device)
        self._buf_reward = torch.empty((self.num_envs,), dtype=torch.float32, device=self.device)

        # targets buffer (Dmax inferred from padded sampler params)
        Pmax = int(self.spec.sampler_kernel_param.shape[1])
        assert Pmax % 2 == 0, "sampler_kernel_param must be padded to even length"
        self._target_dim_max = Pmax // 2

        self._targets = torch.zeros(
            (self.num_envs, self.num_subtasks, self._target_dim_max),
            dtype=torch.float32,
            device=self.device,
        )

        # initialize targets once
        self._resample_targets(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    # --------------------------------------------------------------------- #
    # Spec building
    # --------------------------------------------------------------------- #

    def _subtask_signature(self, subtask_cfg) -> tuple:
        """Dedup signature: MUST include everything that changes behavior."""
        # resolve first so ids are available even if user specified names
        subtask_cfg.asset_cfg.resolve(self._env.scene)

        asset = subtask_cfg.asset_cfg
        body_sig = _ids_sig(asset.body_ids)
        joint_sig = _ids_sig(asset.joint_ids)

        sampler_sig = (
            int(subtask_cfg.sampler.kernel),
            tuple(map(float, subtask_cfg.sampler.minimum)),
            tuple(map(float, subtask_cfg.sampler.maximum)),
        )

        return (
            type(subtask_cfg).__name__,
            asset.name,
            body_sig,
            joint_sig,
            int(subtask_cfg.state_kernel),
            int(subtask_cfg.metric_kernel),
            sampler_sig,
            int(subtask_cfg.activation_kernel),
            float(subtask_cfg.activation_kernel_param),
        )

    def _build_spec(self) -> TaskSpec:
        from .commands_cfg import MultiTaskCfg

        cfg: MultiTaskCfg = self.cfg
        device = self.device

        task_names = list(cfg.tasks.keys())
        signature_to_subtask_id: dict[tuple, int] = {}

        # per-subtask lists
        state_kernel_id: list[int] = []
        metric_kernel_id: list[int] = []
        sampler_kernel_id: list[int] = []
        sampler_kernel_param_rows: list[torch.Tensor] = []
        sampler_kernel_param_len: list[int] = []
        activation_kernel_id: list[int] = []
        activation_kernel_param: list[float] = []
        is_tracking: list[bool] = []
        is_instant: list[bool] = []
        subtask_asset_cfgs: list[SceneEntityCfg] = []

        # for correctness: group by resolved entity signature (name + ids)
        entity_sig_to_id: dict[tuple, int] = {}
        subtask_entity_id: list[int] = []
        task_to_subtask_ids: list[list[int]] = []

        for task_name in task_names:
            row: list[int] = []
            for subtask_cfg in cfg.tasks[task_name]:
                sig = self._subtask_signature(subtask_cfg)
                sid = signature_to_subtask_id.get(sig)

                if sid is None:
                    sid = len(state_kernel_id)
                    signature_to_subtask_id[sig] = sid

                    asset_cfg = subtask_cfg.asset_cfg  # already resolved in signature
                    subtask_asset_cfgs.append(asset_cfg)

                    # entity group id
                    ent_sig = (asset_cfg.name, _ids_sig(asset_cfg.body_ids), _ids_sig(asset_cfg.joint_ids))
                    ent_id = entity_sig_to_id.get(ent_sig)
                    if ent_id is None:
                        ent_id = len(entity_sig_to_id)
                        entity_sig_to_id[ent_sig] = ent_id
                    subtask_entity_id.append(ent_id)

                    state_kernel_id.append(int(subtask_cfg.state_kernel))
                    metric_kernel_id.append(int(subtask_cfg.metric_kernel))

                    sampler_kernel_id.append(int(subtask_cfg.sampler.kernel))
                    prow = subtask_cfg.sampler.get_kernel_input(device=device)  # 1D tensor
                    sampler_kernel_param_rows.append(prow)
                    sampler_kernel_param_len.append(int(prow.numel()))

                    activation_kernel_id.append(int(subtask_cfg.activation_kernel))
                    activation_kernel_param.append(float(subtask_cfg.activation_kernel_param))

                    is_tracking.append(isinstance(subtask_cfg, MultiTaskCfg.TrackingTaskCfg))
                    is_instant.append(isinstance(subtask_cfg, MultiTaskCfg.InstantaneousTaskCfg))

                row.append(sid)

            if not row:
                raise ValueError(f"Task '{task_name}' has no subtasks.")
            task_to_subtask_ids.append(row)

        # pad task->subtask table
        task_subtask_ids, task_subtask_valid = pad_index_rows(task_to_subtask_ids, device=device)

        # pad sampler params to rectangular [S, Pmax]
        Pmax = max(sampler_kernel_param_len) if sampler_kernel_param_len else 0
        if Pmax % 2 == 1:
            Pmax += 1  # keep even so sampler can reshape into pairs

        sampler_param_t = torch.zeros((len(sampler_kernel_param_rows), Pmax), dtype=torch.float32, device=device)
        for j, prow in enumerate(sampler_kernel_param_rows):
            sampler_param_t[j, : prow.numel()] = prow

        return TaskSpec(
            task_names=task_names,
            task_subtask_ids=task_subtask_ids,
            task_subtask_valid=task_subtask_valid,
            state_kernel_id=torch.tensor(state_kernel_id, dtype=torch.long, device=device),
            metric_kernel_id=torch.tensor(metric_kernel_id, dtype=torch.long, device=device),
            sampler_kernel_id=torch.tensor(sampler_kernel_id, dtype=torch.long, device=device),
            sampler_kernel_param=sampler_param_t,
            sampler_kernel_param_len=torch.tensor(sampler_kernel_param_len, dtype=torch.long, device=device),
            activation_kernel_id=torch.tensor(activation_kernel_id, dtype=torch.long, device=device),
            activation_kernel_param=torch.tensor(activation_kernel_param, dtype=torch.float32, device=device),
            is_tracking=torch.tensor(is_tracking, dtype=torch.bool, device=device),
            is_instant=torch.tensor(is_instant, dtype=torch.bool, device=device),
            subtask_asset_cfgs=subtask_asset_cfgs,
            subtask_entity_id=torch.tensor(subtask_entity_id, dtype=torch.long, device=device),
        )

    # --------------------------------------------------------------------- #
    # Resampling
    # --------------------------------------------------------------------- #

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def resample_indices(self, env_ids: torch.Tensor) -> None:
        self.task_samples[env_ids] = torch.randint(0, self.num_tasks, (env_ids.numel(),), device=self.device)

    def _resample_command(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.resample_indices(env_ids)
        self._resample_targets(env_ids)

    def _resample_targets(self, env_ids: torch.Tensor) -> None:
        spec = self.spec
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        task_idx = self.task_samples[env_ids]                 # [num_envs]
        subtask_ids = spec.task_subtask_ids[task_idx]         # [num_envs, num_subtasks]
        valid = spec.task_subtask_valid[task_idx]             # [num_envs, num_subtasks]

        safe = subtask_ids.clamp_min(0)                       # [num_envs, num_subtasks]
        sampler_id = spec.sampler_kernel_id[safe]             # [num_envs, num_subtasks]
        params = spec.sampler_kernel_param[safe]              # [num_envs, num_subtasks, input_max]

        env_grid = env_ids[:, None].expand_as(safe)           # [num_envs, num_subtasks]

        for sid, sampler_fn in enumerate(SAMPLER_KERNELS):
            mask = valid & (sampler_id == sid)
            if not mask.any():
                continue

            env_flat = env_grid[mask]                         # [K]
            subtask_flat = safe[mask]                         # [K]
            params_flat = params[mask]                        # [K, Pmax]

            target_flat = sampler_fn(params_flat)
            self._targets[env_flat, subtask_flat, :self._target_dim_max] = target_flat

    # --------------------------------------------------------------------- #
    # Error + reward
    # --------------------------------------------------------------------- #

    def _compute_error(self) -> torch.Tensor:
        error = torch.zeros((self.num_envs, self.num_subtasks), device=self.device)

        for state_index, state_fn in enumerate(STATE_KERNELS):
            mask_state = (self.spec.state_kernel_id == state_index)
            if not mask_state.any():
                continue

            entity_ids = self.spec.subtask_entity_id[mask_state].unique()
            for ent_id in entity_ids.tolist():  # this for loop can be fixed by newton where asset can be batch selected through indices
                mask_ent = mask_state & (self.spec.subtask_entity_id == ent_id)
                if not mask_ent.any():
                    continue

                j0 = int(mask_ent.nonzero(as_tuple=False)[0])
                asset_cfg = self.spec.subtask_asset_cfgs[j0]

                x_cur = state_fn(self._env, slice(None), asset_cfg)     # [N, ...]
                tail = x_cur.shape[1:]
                dim_x_current = int(torch.tensor(tail).prod().item()) if len(tail) else 1
                if dim_x_current > self._target_dim_max:
                    raise ValueError(f"x_cur dim {dim_x_current} exceeds target dim max {self._target_dim_max}. Fix sampler padding/config.")

                for metric_index, metric_fn in enumerate(METRIC_KERNELS):
                    mask_group = mask_ent & (self.spec.metric_kernel_id == metric_index)
                    if not mask_group.any():
                        continue

                    kernel_shared_subtask_indices = mask_group.nonzero(as_tuple=False).squeeze(-1)
                    num_subtasks_sharing_kernel = kernel_shared_subtask_indices.numel()

                    x_cur_group = x_cur.unsqueeze(1).expand(self.num_envs, num_subtasks_sharing_kernel, *tail)
                    x_tgt_flat = self._targets[:, kernel_shared_subtask_indices, :dim_x_current]
                    x_tgt_group = x_tgt_flat.view(self.num_envs, num_subtasks_sharing_kernel, *tail)

                    err = metric_fn(x_cur_group, x_tgt_group)
                    while err.dim() > 2:
                        err = err.mean(dim=-1)

                    error[:, kernel_shared_subtask_indices] = err

        return error

    def _apply_activation_kernels(self, error: torch.Tensor) -> torch.Tensor:
        activated = torch.empty_like(error)
        for kid, kfn in enumerate(ACTIVATION_KERNELS):
            mask = self.spec.activation_kernel_id == kid
            if not mask.any():
                continue
            param = self.spec.activation_kernel_param[mask]
            out = kfn(error[:, mask], param)
            activated[:, mask] = out.to(error.dtype)
        return activated

    def _select_subtasks(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_subtasks = self.spec.task_subtask_ids.shape[1]
        safe_ids = self._buf_safe_ids[:, :max_subtasks]
        selected = self.spec.task_subtask_ids[self.task_samples]
        safe_ids.copy_(selected)
        safe_ids.clamp_(min=0)
        valid = self.spec.task_subtask_valid[self.task_samples]
        return safe_ids, valid

    @staticmethod
    def _compute_tracking_mean(values: torch.Tensor, is_tracking: torch.Tensor) -> torch.Tensor:
        cnt = is_tracking.sum(dim=1).clamp_min(1)
        return (values * is_tracking).sum(dim=1) / cnt

    def _compute_reach_success(self, values: torch.Tensor, is_instant: torch.Tensor) -> torch.Tensor:
        N, M = values.shape
        inst = self._buf_instant_factor[:N, :M]
        inst.copy_(values)
        inst[~is_instant] = 1.0
        return inst.prod(dim=1)

    def get_task_reward(self) -> torch.Tensor:
        spec = self.spec
        N = self.num_envs
        M = spec.task_subtask_ids.shape[1]

        error = self._compute_error()
        activated = self._apply_activation_kernels(error)

        safe_ids, selected_valid = self._select_subtasks()

        selected_value = self._buf_selected_value[:N, :M]
        selected_value.copy_(activated.gather(1, safe_ids))
        selected_value.masked_fill_(~selected_valid, 0.0)

        is_tracking = self._buf_is_tracking[:N, :M]
        is_instant = self._buf_is_instant[:N, :M]
        is_tracking.copy_(spec.is_tracking[safe_ids])
        is_instant.copy_(spec.is_instant[safe_ids])
        is_tracking &= selected_valid
        is_instant &= selected_valid

        has_tracking = is_tracking.any(dim=1)
        has_instant = is_instant.any(dim=1)

        tracking_mean = self._compute_tracking_mean(selected_value, is_tracking)
        tracking_reward = ((1.0 - self.gamma) / self.max_episode_length) * tracking_mean

        reach_success = self._compute_reach_success(selected_value, is_instant)

        episode_step = self._env.episode_length_buf.to(torch.float32)
        ramp = 1.0 - episode_step / self.max_episode_length
        mixed_reward = (1.0 / self.max_episode_length) * tracking_mean + reach_success * (1.0 + ramp * tracking_mean)

        reward = self._buf_reward[:N]
        reward.zero_()
        reward[has_tracking & ~has_instant] = tracking_reward[has_tracking & ~has_instant]
        reward[~has_tracking & has_instant] = reach_success[~has_tracking & has_instant]
        reward[has_tracking & has_instant] = mixed_reward[has_tracking & has_instant]
        return reward

    def get_task_done(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
