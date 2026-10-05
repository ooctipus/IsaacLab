# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success-rate monitoring shared by task reset strategies."""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

from isaaclab.utils import configclass


@wp.kernel
def _append_success(
    slots: wp.array(dtype=Any),
    success: wp.array(dtype=Any),
    valid: wp.array(dtype=bool),
    order: wp.array(dtype=wp.int64),
    outcomes: wp.array2d(dtype=float),
    pointers: wp.array(dtype=wp.int64),
    sizes: wp.array(dtype=wp.int64),
    rates: wp.array(dtype=float),
    valid_indices: wp.array(dtype=int),
):
    end = wp.tid()
    slot = slots[order[end]]
    if slot < 0 or slot >= outcomes.shape[0]:
        if valid.shape[0] == 0 or valid[order[end]]:
            wp.atomic_min(valid_indices, 0, 0)
        return
    if end + 1 < order.shape[0] and slots[order[end + 1]] == slots[order[end]]:
        return
    # Stable sorting assigns one writer to each slot and preserves episode order.
    # Only the last history-sized suffix matters, even for a full-population reset.
    history = outcomes.shape[1]
    begin = end
    written = int(0)
    while begin >= 0 and slots[order[begin]] == slot and written < history:
        if valid.shape[0] == 0 or valid[order[begin]]:
            written += 1
        begin -= 1
    if written == 0:
        return
    pointer = int(pointers[slot])
    for entry in range(begin + 1, end + 1):
        row = order[entry]
        if valid.shape[0] == 0 or valid[row]:
            outcomes[slot, pointer] = float(success[row])
            pointer = (pointer + 1) % history
    pointers[slot] = wp.int64(pointer)
    sizes[slot] = wp.min(sizes[slot] + wp.int64(written), wp.int64(history))
    total = float(0.0)
    for column in range(history):
        total += outcomes[slot, column]
    rates[slot] = total / float(sizes[slot])


@configclass
class SuccessMonitorCfg:
    """Configuration for :class:`SuccessMonitor`."""

    class_type: type[SuccessMonitor] | str = "{DIR}.success_monitor:SuccessMonitor"
    """Monitor implementation, resolved when the environment starts."""

    monitored_history_len: int = 10
    """Episodes remembered per slot."""

    target_success_rate: float = 0.5
    """Success rate favored by sampling, in ``[0, 1]``."""

    kappa: float = 1.0
    """Concentration around :attr:`target_success_rate`; zero is uniform."""

    temperature: float = 1.0
    """Sampling-weight temperature, at or above ``1.0``."""


class SuccessMonitor:
    """Track recent outcomes per slot and sample within partitioned slot banks."""

    def __init__(self, cfg: SuccessMonitorCfg, num_partitions: int, partition_size: int, device: str):
        if cfg.monitored_history_len < 1:
            raise ValueError("Success history must remember at least one episode.")
        wp.init()
        self.cfg = cfg
        self.num_partitions = num_partitions
        self.partition_size = partition_size
        self.device = device

        num_slots = num_partitions * partition_size
        self.success_buf = torch.zeros((num_slots, cfg.monitored_history_len), dtype=torch.float32, device=device)
        self.success_rate = torch.zeros(num_slots, dtype=torch.float32, device=device)
        self.success_pointer = torch.zeros(num_slots, device=device, dtype=torch.long)
        self.success_size = torch.zeros(num_slots, device=device, dtype=torch.long)

    def get_success_rate(self) -> torch.Tensor:
        """Return a copy of every slot's measured success rate."""
        return self.success_rate.clone()

    def get_mean_success_rate(self) -> torch.Tensor:
        """Average rates across slots that have recorded outcomes, as a 0-d device tensor; zero before any outcome."""
        measured = self.success_size > 0
        return (self.success_rate * measured).sum() / measured.sum().clamp(min=1)

    def success_update(self, slot_ids: torch.Tensor, success: torch.Tensor, *, valid: torch.Tensor | None = None):
        """Append outcomes in input order and update only the affected slots' rates.

        ``valid`` optionally excludes entries without compacting the input. Excluded
        slot IDs are never dereferenced; included IDs must index this monitor's bank.
        Inputs are equally sized vectors on the monitor device. If a batch contains
        more than one history of outcomes for a slot, only its last history is appended.
        """
        if (
            slot_ids.ndim != 1
            or slot_ids.dtype not in (torch.int32, torch.int64)
            or success.shape != slot_ids.shape
            or success.is_complex()
            or slot_ids.device != self.success_buf.device
            or success.device != slot_ids.device
        ):
            raise ValueError(
                "Success updates require integer slot IDs and equally sized numeric outcomes on the bank device."
            )
        if valid is not None and (
            valid.shape != slot_ids.shape or valid.dtype != torch.bool or valid.device != slot_ids.device
        ):
            raise ValueError("Success validity must be an equally sized boolean vector on the bank device.")
        if len(slot_ids) == 0:
            return
        order = torch.argsort(slot_ids, stable=True)
        valid_indices = torch.ones(1, dtype=torch.int32, device=slot_ids.device)
        wp.launch(
            _append_success,
            dim=len(slot_ids),
            inputs=[
                wp.from_torch(slot_ids),
                wp.from_torch(success),
                None if valid is None else wp.from_torch(valid),
                wp.from_torch(order),
                wp.from_torch(self.success_buf),
                wp.from_torch(self.success_pointer),
                wp.from_torch(self.success_size),
                wp.from_torch(self.success_rate),
                wp.from_torch(valid_indices),
            ],
            device=self.device,
            stream=wp.stream_from_torch(torch.cuda.current_stream(self.device)) if self.success_buf.is_cuda else None,
        )
        torch._assert_async(valid_indices, "Included success slot IDs must index the monitor bank.")

    def target_weights(self) -> torch.Tensor:
        """Return unnormalized slot weights peaking at the target success rate."""
        target = min(max(self.cfg.target_success_rate, 0.0), 1.0)
        kappa = max(self.cfg.kappa, 0.0)
        a = 1.0 + kappa * target
        b = 1.0 + kappa * (1.0 - target)
        eps = 1e-4
        rate = self.success_rate
        weights = ((rate + eps).pow(a - 1.0) * (1.0 - rate + eps).pow(b - 1.0)).clamp_min(eps)
        return weights.pow(1.0 / max(self.cfg.temperature, 1.0))

    def sample_by_target_rate(self, partition_ids: torch.Tensor) -> torch.Tensor:
        """Draw one slot from each requested partition."""
        weights = self.target_weights().view(self.num_partitions, self.partition_size)
        slots = torch.multinomial(weights[partition_ids], 1).view(-1)
        return partition_ids * self.partition_size + slots
