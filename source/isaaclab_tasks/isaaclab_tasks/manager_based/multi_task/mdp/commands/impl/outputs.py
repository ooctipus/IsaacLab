# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-owned command output storage layouts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..multi_task_command import MultiTaskCommand


class DenseCommandOutputs:
    """Dense ``[env, slot]`` and canonical command output layout.

    This is the legacy layout used by the reference, ``mega_kernel``, and
    ``packed_scatter`` backends. It lives behind the backend/output boundary so
    future backends can replace the hot-path layout without forcing every
    implementation to materialize dense slot tensors internally.
    """

    def __init__(self, command: MultiTaskCommand) -> None:
        device = command.device
        num_envs = command.num_envs
        k_max = command.k_max
        self.buf_error = torch.zeros((num_envs, k_max), device=device)
        self.buf_activation = torch.zeros((num_envs, k_max), device=device)
        self.command_reach = torch.zeros((num_envs, max(1, command.spec.reach_canonical_width)), device=device)
        self.command_track = torch.zeros((num_envs, max(1, command.spec.track_canonical_width)), device=device)
        self.task_reward = torch.zeros(num_envs, device=device)
        self.task_done_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.progress = torch.zeros(num_envs, device=device)

    @property
    def command(self) -> torch.Tensor:
        """Concatenated reach + track command tensor."""
        return torch.cat([self.command_reach, self.command_track], dim=-1)

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        """Reset per-env output rows after task resampling."""
        self.task_reward[env_ids] = 0.0
        self.task_done_success[env_ids] = False
        self.progress[env_ids] = 0.0
        self.command_reach[env_ids] = 0.0
        self.command_track[env_ids] = 0.0

    def reset_step(self, valid_slots: torch.Tensor) -> None:
        """Clear per-step dense slot outputs before dispatch."""
        del valid_slots
        self.buf_error.zero_()
        self.buf_activation.zero_()


class PrimitiveLocalCommandOutputs(DenseCommandOutputs):
    """Local queued output layout for ``primitive_queue_local``.

    The policy-facing canonical command tensors and debug dense slot tensors
    are still present at the public boundary. The backend hot path owns
    contiguous local rows and composes reward through ``slot_local_index``.
    """

    def __init__(self, command: MultiTaskCommand) -> None:
        super().__init__(command)
        max_work = command.num_envs * command.k_max
        self.local_delta = torch.zeros((max_work, 4), device=command.device)
        self.local_error = torch.zeros(max_work, device=command.device)
        self.local_activation = torch.zeros(max_work, device=command.device)
        self.slot_local_index = torch.zeros((command.num_envs, command.k_max), device=command.device, dtype=torch.int32)

    def reset_step(self, valid_slots: torch.Tensor) -> None:
        """Clear debug dense slot tensors before dispatch.

        Local rows are overwritten by the primitive queue dispatch. The dense
        clear keeps current metrics/tests compatible while the composer reads
        from ``local_activation``.
        """
        del valid_slots
        self.buf_error.zero_()
        self.buf_activation.zero_()
