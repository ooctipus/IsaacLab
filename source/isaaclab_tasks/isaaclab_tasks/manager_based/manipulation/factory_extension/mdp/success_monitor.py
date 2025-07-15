from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .success_monitor_cfg import SuccessMonitorCfg, SequentialDataSuccessMonitorCfg


class SuccessMonitor:
    def __init__(self, cfg: SuccessMonitorCfg):

        # uniform success buff
        self.monitored_history_len = cfg.monitored_history_len
        self.device = cfg.device
        self.success_buf = torch.zeros((cfg.num_monitored_data, self.monitored_history_len), device=self.device)
        self.success_rate = torch.zeros((cfg.num_monitored_data), device=self.device)
        self.success_pointer = torch.zeros((cfg.num_monitored_data), device=self.device, dtype=torch.int32)
        self.success_size = torch.zeros((cfg.num_monitored_data), device=self.device, dtype=torch.int32)

    def failure_rate_sampling(self, env_ids):
        failure_rate = (1 - self.success_rate).clamp(min=1e-6)
        return torch.multinomial(failure_rate.view(-1), len(env_ids), replacement=True).to(torch.int32)

    def success_update(self, ids_all, success_mask):
        unique_indices, inv, counts = torch.unique(ids_all, return_inverse=True, return_counts=True)
        counts_clamped = counts.clamp(max=self.monitored_history_len).to(dtype=self.success_pointer.dtype)

        ptrs = self.success_pointer[unique_indices]
        values = (success_mask[torch.argsort(inv)]).to(device=self.device, dtype=self.success_buf.dtype)
        values_splits = torch.split(values, counts.tolist())
        clamped_values = torch.cat([grp[-n:] for grp, n in zip(values_splits, counts_clamped.tolist())])
        state_indices = torch.repeat_interleave(unique_indices, counts_clamped)
        buf_indices = torch.cat([
            torch.arange(start, start + n, dtype=torch.int64, device=self.device) % self.monitored_history_len
            for start, n in zip(ptrs.tolist(), counts_clamped.tolist())
        ])

        self.success_buf.index_put_((state_indices, buf_indices), clamped_values)

        self.success_pointer.index_add_(0, unique_indices, counts_clamped)
        self.success_pointer = self.success_pointer % self.monitored_history_len

        self.success_size.index_add_(0, unique_indices, counts_clamped)
        self.success_size = self.success_size.clamp(max=self.monitored_history_len)
        self.success_rate[:] = self.success_buf.sum(dim=1) / self.success_size.clamp(min=1)

    def get_success_rate(self):
        return self.success_rate.clone()


class SequentialDataSuccessMonitor(SuccessMonitor):
    def __init__(self, cfg: SequentialDataSuccessMonitorCfg):
        super().__init__(cfg)
        self.success_rate_report_chunk_size = cfg.success_rate_report_chunk_size
        self.episode_len = torch.tensor(cfg.episode_length_list, device=self.device)
        self.num_episodes = len(self.episode_len)
        self.max_episode_length = self.episode_len.max().item()

        zeros = torch.arange(self.max_episode_length, device=self.device)[None, :] < self.episode_len[:, None]
        self.oob_mask = (~zeros).reshape(-1)
        oob_mask_expanded = self.oob_mask.unsqueeze(1).expand(self.success_buf.shape)
        self.success_buf[oob_mask_expanded] = 1
        self.success_rate[self.oob_mask] = 1
        self.success_size[self.oob_mask] = self.monitored_history_len

        # create one-hot bin for success interval calculation
        self.col_indices = torch.arange(self.max_episode_length, device=self.device).repeat(self.num_episodes, 1).float()
        normalized_steps = self.col_indices / (self.episode_len.unsqueeze(1) - 1)
        normalized_steps[:, 0] += 1e-5
        bin_edges = torch.linspace(0, 1, self.success_rate_report_chunk_size + 1, device=self.device)
        bin_indices = torch.bucketize(normalized_steps, bin_edges, right=False) - 1
        bin_indices = torch.where((~self.oob_mask), bin_indices.reshape(-1), -1)
        self.one_hot_bins = F.one_hot(bin_indices.clamp(min=0), num_classes=self.success_rate_report_chunk_size).to(torch.int32)
        self.one_hot_bins = self.one_hot_bins * (~self.oob_mask).unsqueeze(-1)
        self.one_hot_bin_counts = self.one_hot_bins.sum(dim=0).clamp(min=1)

    # Used for episode based data
    def get_success_rate_bin(self):
        weighted_success = self.success_rate.unsqueeze(-1) * self.one_hot_bins
        overall_bin_success = weighted_success.sum(dim=0) / self.one_hot_bin_counts
        return overall_bin_success