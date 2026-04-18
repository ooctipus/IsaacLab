# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, SceneEntityCfg
from isaaclab.utils.buffers import TimestampedBuffer

from .kernels import ACTIVATION_KERNELS, DELTA_KERNELS, METRIC_KERNELS, SAMPLER_KERNELS, STATE_KERNELS

if TYPE_CHECKING:
    from .commands_cfg import MultiTaskCfg


def kernel_apply(
    kernels: Sequence[Callable],
    kernel_ids: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    out: torch.Tensor,
    out_indexer: Callable[[torch.Tensor], tuple],
    kernel_args: Callable[[torch.Tensor], tuple],
) -> None:
    for kid, kernel_fn in enumerate(kernels):
        mask = (kernel_ids == kid) if valid is None else (valid & (kernel_ids == kid))
        if not mask.any():
            continue
        args = kernel_args(mask)
        assert isinstance(args, tuple), "kernel_args(mask) must return a tuple, e.g. (x,) not x"
        out[out_indexer(mask)] = kernel_fn(*args).to(out.dtype)


def pad_index_rows(index_rows: list[list[int]], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
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
    sampler_kernel_param: torch.Tensor
    activation_kernel_id: torch.Tensor
    activation_kernel_param: torch.Tensor

    is_tracking: torch.Tensor
    is_instant: torch.Tensor

    subtask_asset_cfgs: list[SceneEntityCfg]
    subtask_entity_id: torch.Tensor


class MultiTaskCommand(CommandTerm):
    def __init__(self, cfg: MultiTaskCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.gamma = 0.99
        self.max_episode_length = float(env.max_episode_length)

        self.spec = self._build_spec()
        self.num_tasks = len(self.spec.task_names)
        self.num_subtasks = int(self.spec.state_kernel_id.numel())

        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self.task_samples = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)

        # work buffers
        self._buf_error = TimestampedBuffer()
        self._buf_delta = TimestampedBuffer()
        self._buf_reward = TimestampedBuffer()
        self._buf_done = TimestampedBuffer()

        max_subtasks = self.spec.task_subtask_ids.shape[1]
        self._buf_safe_ids = torch.empty((self.num_envs, max_subtasks), dtype=torch.long, device=self.device)
        self._buf_selected_value = torch.empty((self.num_envs, max_subtasks), dtype=torch.float32, device=self.device)
        self._buf_is_tracking = torch.empty((self.num_envs, max_subtasks), dtype=torch.bool, device=self.device)
        self._buf_is_instant = torch.empty((self.num_envs, max_subtasks), dtype=torch.bool, device=self.device)
        self._buf_instant_factor = torch.empty((self.num_envs, max_subtasks), dtype=torch.float32, device=self.device)
        self._buf_activated = torch.empty((self.num_envs, self.num_subtasks), device=self.device)
        self._buf_reward.data = torch.empty((self.num_envs,), dtype=torch.float32, device=self.device)
        self._buf_done.data = torch.empty((self.num_envs,), dtype=torch.bool, device=self.device)
        self._buf_error.data = torch.empty((self.num_envs, self.num_subtasks), device=self.device)

        Pmax = int(self.spec.sampler_kernel_param.shape[1])
        assert Pmax % 2 == 0, "sampler_kernel_param must be padded to even length"
        self._target_dim_max = Pmax // 2
        self._buf_delta.data = torch.empty((self.num_envs, self.num_subtasks, self._target_dim_max), device=self.device)
        self._targets = torch.zeros((self.num_envs, self.num_subtasks, self._target_dim_max), device=self.device)

        self._validate()
        self._resample_command(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def _validate(self):
        assert len(DELTA_KERNELS) == len(METRIC_KERNELS), "DELTA_KERNELS and METRIC_KERNELS must align by id"
        assert int(self.spec.metric_kernel_id.max().item()) < len(METRIC_KERNELS)
        assert int(self.spec.state_kernel_id.max().item()) < len(STATE_KERNELS)
        assert int(self.spec.activation_kernel_id.max().item()) < len(ACTIVATION_KERNELS)
        assert int(self.spec.sampler_kernel_id.max().item()) < len(SAMPLER_KERNELS)

    def _subtask_signature(self, subtask_cfg: MultiTaskCfg.BaseTaskCfg) -> tuple:
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
        activation_kernel_id: list[int] = []
        activation_kernel_param: list[float] = []
        is_tracking: list[bool] = []
        is_instant: list[bool] = []
        subtask_asset_cfgs: list[SceneEntityCfg] = []

        entity_sig_to_id: dict[tuple, int] = {}
        subtask_entity_id: list[int] = []
        task_to_subtask_ids: list[list[int]] = []
        Pmax = 0
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
                    Pmax = max(Pmax, int(prow.numel()))

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

        # pad sampler params to rectangular [S, Pmax] TODO: this is still a hack, need to figure out a better way
        if Pmax % 2 == 1:
            Pmax += 1

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
            activation_kernel_id=torch.tensor(activation_kernel_id, dtype=torch.long, device=device),
            activation_kernel_param=torch.tensor(activation_kernel_param, dtype=torch.float32, device=device),
            is_tracking=torch.tensor(is_tracking, dtype=torch.bool, device=device),
            is_instant=torch.tensor(is_instant, dtype=torch.bool, device=device),
            subtask_asset_cfgs=subtask_asset_cfgs,
            subtask_entity_id=torch.tensor(subtask_entity_id, dtype=torch.long, device=device),
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def task_done(self) -> torch.Tensor:
        if self._buf_done.timestamp < self.timestamp:
            self._update_reward_done()

        return self._buf_done.data

    @property
    def task_reward(self) -> torch.Tensor:
        if self._buf_reward.timestamp < self.timestamp:
            self._update_reward_done()

        return self._buf_reward.data

    @property
    def buf_delta(self) -> torch.Tensor:
        if self._buf_delta.timestamp < self.timestamp:
            self._update()

        return self._buf_delta.data

    @property
    def buf_error(self) -> torch.Tensor:
        if self._buf_error.timestamp < self.timestamp:
            self._update()

        return self._buf_error.data

    @property
    def timestamp(self):
        return self._env.common_step_counter  # type: ignore

    def _update_command(self):
        self.buf_error

    def _update(self):
        if self._buf_error.timestamp < self.timestamp or self._buf_delta.timestamp < self.timestamp:
            self._buf_error.data.fill_(float("nan"))
            for state_index, state_fn in enumerate(STATE_KERNELS):
                mask_state = self.spec.state_kernel_id == state_index
                if not mask_state.any():
                    continue

                entity_ids = self.spec.subtask_entity_id[mask_state].unique()
                for ent_id in entity_ids.tolist():
                    mask_ent = mask_state & (self.spec.subtask_entity_id == ent_id)
                    if not mask_ent.any():
                        continue

                    j0 = int(mask_ent.nonzero(as_tuple=False)[0])
                    asset_cfg = self.spec.subtask_asset_cfgs[j0]

                    x_cur = state_fn(self._env, slice(None), asset_cfg)  # [env, ...]
                    tail = x_cur.shape[1:]
                    dim_x_cur = math.prod(tail) if len(tail) else 1

                    if dim_x_cur > self._target_dim_max:
                        raise ValueError(
                            f"x_cur dim {dim_x_cur} exceeds target dim max {self._target_dim_max}. "
                            "Fix sampler padding/config."
                        )

                    # Flatten current state to [env, 1, D]
                    x_cur_flat = x_cur.reshape(self.num_envs, dim_x_cur).unsqueeze(1)

                    # ---- (A) compute delta into _buf_delta ----
                    kernel_apply(
                        kernels=DELTA_KERNELS,
                        kernel_ids=self.spec.metric_kernel_id,  # or spec.delta_kernel_id if you add it
                        valid=mask_ent,
                        out=self._buf_delta.data,
                        out_indexer=lambda mask: (slice(None), mask, slice(0, dim_x_cur)),
                        kernel_args=lambda mask: (
                            x_cur_flat.expand(self.num_envs, self._targets[:, mask, :dim_x_cur].shape[1], dim_x_cur),
                            self._targets[:, mask, :dim_x_cur],
                        ),
                    )

                    # ---- (B) reduce delta -> scalar metric into _buf_error ----
                    kernel_apply(
                        kernels=METRIC_KERNELS,
                        kernel_ids=self.spec.metric_kernel_id,
                        valid=mask_ent,
                        out=self._buf_error.data,
                        out_indexer=lambda mask: (slice(None), mask),
                        kernel_args=lambda mask: (self._buf_delta.data[:, mask, :dim_x_cur],),
                    )
            assert not torch.isnan(self._buf_error.data).any(), "Bug some error buffer is not updated."
            self._buf_delta.timestamp = self.timestamp
            self._buf_error.timestamp = self.timestamp

    def _update_metrics(self):
        pass

    def resample_indices(self, env_ids: torch.Tensor) -> None:
        self.task_samples[env_ids] = torch.randint(0, self.num_tasks, (env_ids.numel(),), device=self.device)

    def _resample_command(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        self.resample_indices(env_ids)
        task_idx = self.task_samples[env_ids]
        subtask_ids = self.spec.task_subtask_ids[task_idx].clamp_min(0)
        params = self.spec.sampler_kernel_param[subtask_ids]
        env_grid = env_ids[:, None].expand_as(subtask_ids)

        kernel_apply(
            kernels=SAMPLER_KERNELS,
            kernel_ids=self.spec.sampler_kernel_id[subtask_ids],
            valid=self.spec.task_subtask_valid[task_idx],
            out=self._targets,
            out_indexer=lambda ker_mask: (env_grid[ker_mask], subtask_ids[ker_mask], slice(0, self._target_dim_max)),
            kernel_args=lambda ker_mask: (params[ker_mask],),
        )

        self._buf_error.timestamp = -1
        self._buf_delta.timestamp = -1
        self._buf_done.timestamp = -1
        self._buf_reward.timestamp = -1

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

    def _update_reward_done(self) -> None:
        if self._buf_reward.timestamp >= self.timestamp and self._buf_done.timestamp >= self.timestamp:
            return

        max_num_subtask = self.spec.task_subtask_ids.shape[1]
        _ = self.buf_error  # ensure error is fresh (lazy)

        # 1) activated[subtask] for all envs
        kernel_apply(
            ACTIVATION_KERNELS,
            self.spec.activation_kernel_id,
            out=self._buf_activated,
            out_indexer=lambda mask: (slice(None), mask),
            kernel_args=lambda mask: (self.buf_error[:, mask], self.spec.activation_kernel_param[mask]),
        )

        # 2) pick subtasks for each env’s sampled task
        safe_ids, selected_valid = self._select_subtasks()

        selected_value = self._buf_selected_value[: self.num_envs, :max_num_subtask]
        selected_value.copy_(self._buf_activated.gather(1, safe_ids))
        selected_value.masked_fill_(~selected_valid, 0.0)

        is_tracking = self._buf_is_tracking[: self.num_envs, :max_num_subtask]
        is_instant = self._buf_is_instant[: self.num_envs, :max_num_subtask]
        is_tracking.copy_(self.spec.is_tracking[safe_ids])
        is_instant.copy_(self.spec.is_instant[safe_ids])
        is_tracking &= selected_valid
        is_instant &= selected_valid

        has_tracking = is_tracking.any(dim=1)
        has_instant = is_instant.any(dim=1)

        # DONE
        instant_ok = selected_value > 0.5  # [env, M] bool
        instant_ok |= ~is_instant  # treat non-instant as "don't care"
        reach_success_bool = instant_ok.all(dim=1)  # [env]
        done = has_instant & reach_success_bool  # [env]
        self._buf_done.data[: self.num_envs] = done

        # REWARD
        tracking_mean = self._compute_tracking_mean(selected_value, is_tracking)
        tracking_reward = ((1.0 - self.gamma) / self.max_episode_length) * tracking_mean

        reach_success = self._compute_reach_success(selected_value, is_instant)  # float 0/1 for boolean kernels

        episode_step = self._env.episode_length_buf.to(torch.float32)
        ramp = 1.0 - episode_step / self.max_episode_length
        mixed_reward = (1.0 / self.max_episode_length) * tracking_mean + reach_success * (1.0 + ramp * tracking_mean)

        reward = self._buf_reward.data[: self.num_envs]
        reward.fill_(float("nan"))
        reward[has_tracking & ~has_instant] = tracking_reward[has_tracking & ~has_instant]
        reward[~has_tracking & has_instant] = reach_success[~has_tracking & has_instant]
        reward[has_tracking & has_instant] = mixed_reward[has_tracking & has_instant]
        assert not torch.isnan(reward).any(), "Bug: reward buffer not fully assigned."

        # stamp both
        self._buf_reward.timestamp = self.timestamp
        self._buf_done.timestamp = self.timestamp
