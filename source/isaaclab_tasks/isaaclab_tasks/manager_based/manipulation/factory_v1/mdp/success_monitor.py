from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .success_monitor import SuccessMonitorCfg, SequentialDataSuccessMonitorCfg


class SuccessMonitor:
    def __init__(self, cfg: SuccessMonitorCfg):

        # uniform success buff
        self.monitored_history_len = cfg.monitored_history_len
        self.device = cfg.device
        self.success_buf = torch.zeros((cfg.num_monitored_data, self.monitored_history_len), device=self.device)
        self.success_rate = torch.zeros((cfg.num_monitored_data), device=self.device)
        self.success_pointer = torch.zeros((cfg.num_monitored_data), device=self.device, dtype=torch.int32)
        self.success_size = torch.zeros((cfg.num_monitored_data), device=self.device, dtype=torch.int32)

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

    def sample_by_target_rate(
        self,
        env_ids: torch.Tensor,
        target: float = 0.5,
        kappa: float = 2.0,
        return_probs: bool = False,
        temperature: float = 2.0,
    ):
        """
        Sample partitions preferring success rates near `target` in [0, 1].

        Weight ~ Beta(a, b) shape on p, with:
            a = 1 + kappa * target
            b = 1 + kappa * (1 - target)

        Special cases:
          - target=0, kappa=1  => w ∝ (1 - p) (failure-focused, like before)
          - target=1, kappa=1  => w ∝ p       (success-focused)
          - target=0.5, kappa=2 => w ∝ p(1 - p) (balanced around 0.5)

        Args:
            env_ids: environments to draw assignments for (length = batch size)
            target: desired success rate peak in [0, 1]
            kappa: concentration (sharpness). Larger -> tighter around `target`.
            return_probs: also return the normalized probs used for sampling.

        Returns:
            choices (int32 indices) [len(env_ids)]
            (optionally) probs [num_partitions]
        """
        p = self.success_rate  # [num_partitions], float
        t = float(max(0.0, min(1.0, target)))
        k = float(max(0.0, kappa))

        # Beta-like shape on p with mode near `t`
        # a,b >= 1 ensures nonnegative exponents even at edges; interior mode if a,b>1.
        a = 1.0 + k * t
        b = 1.0 + k * (1.0 - t)

        eps = 1e-8  # avoids 0^0 and zero-sum
        w = ((p + eps).pow(a - 1.0) * (1.0 - p + eps).pow(b - 1.0)).clamp_min(eps)

        logits = torch.log(w + eps)
        probs = torch.softmax(logits / max(1.0, float(temperature)), dim=0)
        choices = torch.multinomial(probs, len(env_ids), replacement=True).to(torch.int32)
        return (choices, probs) if return_probs else choices
