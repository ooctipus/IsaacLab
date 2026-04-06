# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .success_monitor_cfg import SuccessMonitorCfg


class SuccessMonitor:
    def __init__(self, cfg: SuccessMonitorCfg):

        self.monitored_history_len = cfg.monitored_history_len
        self.device = cfg.device
        n = cfg.num_monitored_data
        self.success_buf = torch.zeros((n, self.monitored_history_len), device=self.device)
        self.success_rate = torch.zeros(n, device=self.device)
        self.success_pointer = torch.zeros(n, device=self.device, dtype=torch.int32)
        self.success_size = torch.zeros(n, device=self.device, dtype=torch.int32)

        self.tag_names: list[str] | None = cfg.tag_names
        self.tags = torch.full((n,), -1, device=self.device, dtype=torch.int64)
        self._tag_counts_dirty = True
        self._tag_counts: torch.Tensor | None = None
        self._tag_has_data: torch.Tensor | None = None
        self._tag_valid: torch.Tensor | None = None
        self._tag_valid_idx: torch.Tensor | None = None
        num_tags = len(self.tag_names) if self.tag_names else 0
        self._tag_sums = torch.zeros(num_tags, device=self.device)

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
        self.success_pointer.remainder_(self.monitored_history_len)

        self.success_size.index_add_(0, unique_indices, counts_clamped)
        self.success_size.clamp_(max=self.monitored_history_len)
        self.success_rate[:] = self.success_buf.sum(dim=1) / self.success_size.clamp(min=1)

    def set_tag_names(self, tag_names: list[str]):
        self.tag_names = tag_names
        self._tag_sums = torch.zeros(len(tag_names), device=self.device)

    def set_tags(self, indices: torch.Tensor, tag_ids: torch.Tensor):
        self.tags[indices] = tag_ids.long()
        self._tag_counts_dirty = True

    def get_success_rate(self):
        return self.success_rate.clone()

    def get_tagged_success_rate(self) -> dict[str, float]:
        if self.tag_names is None:
            return {}
        num_tags = len(self.tag_names)
        if self._tag_counts_dirty:
            self._tag_valid = self.tags >= 0
            self._tag_valid_idx = self.tags[self._tag_valid]
            self._tag_counts = torch.bincount(self._tag_valid_idx, minlength=num_tags).float()
            self._tag_has_data = self._tag_counts > 0
            self._tag_counts_dirty = False
        self._tag_sums.zero_()
        self._tag_sums.scatter_add_(0, self._tag_valid_idx, self.success_rate[self._tag_valid])
        means = self._tag_sums / self._tag_counts.clamp(min=1)
        return {name: means[i].item() for i, name in enumerate(self.tag_names) if self._tag_has_data[i]}

    def sample_by_target_rate(
        self,
        env_ids: torch.Tensor,
        target: float = 0.5,
        kappa: float = 2.0,
        return_probs: bool = False,
        temperature: float = 2.0,
    ):
        """Sample partitions preferring success rates near ``target``.

        Args:
            env_ids: Environments to draw assignments for.
            target: Desired success rate peak in [0, 1].
            kappa: Concentration around target.
            return_probs: Also return the sampling probabilities.
            temperature: Softmax temperature.

        Returns:
            Indices tensor ``(len(env_ids),)`` and optionally probabilities.
        """
        p = self.success_rate
        t = float(max(0.0, min(1.0, target)))
        k = float(max(0.0, kappa))
        a = 1.0 + k * t
        b = 1.0 + k * (1.0 - t)

        eps = 1e-8
        w = ((p + eps).pow(a - 1.0) * (1.0 - p + eps).pow(b - 1.0)).clamp_min(eps)
        probs = torch.softmax(torch.log(w + eps) / max(1.0, float(temperature)), dim=0)
        choices = torch.multinomial(probs, len(env_ids), replacement=True).to(torch.int32)
        return (choices, probs) if return_probs else choices
