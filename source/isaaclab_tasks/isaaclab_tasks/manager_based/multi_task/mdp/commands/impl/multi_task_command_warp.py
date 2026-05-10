# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public Warp-backed :class:`MultiTaskCommand` subclass.

This file is intentionally the public switchboard only. The actual backend
pieces live under :mod:`.impl`:

``mega_kernel``, ``schedule_ordered_mega``, ``packed_scatter``,
``primitive_queue_local``, and ``primitive_graph_local`` are wired today. Each
backend owns its execution plan under :mod:`.impl`.

Selected when ``MultiTaskCfg.dispatch_backend`` is not ``"torch"``. The factory
in :class:`~..multi_task_command.MultiTaskCommand.__new__` routes construction
to this class automatically; users never reference it directly.
"""

from __future__ import annotations

import torch

from ..multi_task_command import MultiTaskCommand
from . import CommandBackend, build_command_backend, build_command_output_store

__all__ = ["MultiTaskCommandWarp"]


# ---------------------------------------------------------------------------
# Subclass.
# ---------------------------------------------------------------------------


class MultiTaskCommandWarp(MultiTaskCommand):
    """Warp-native command term.

    Public IsaacLab integration stays here; backend implementation details are
    split by phase in :mod:`.impl`.
    """

    def __init__(self, cfg, env):
        self._backend: CommandBackend | None = None
        super().__init__(cfg, env)
        self._backend = build_command_backend(self, cfg.dispatch_backend)

    def _build_output_store(self):
        """Create the output storage layout required by the selected backend."""
        return build_command_output_store(self, self.cfg.dispatch_backend)

    def _on_resample_command(self, env_ids: torch.Tensor) -> None:
        """Refresh backend-owned execution plans after task assignment changes."""
        if self._backend is not None:
            self._backend.on_resample(self, env_ids)

    def _update_command(self) -> None:
        """Per-step Warp update — skip the Torch overhead the base class adds for the Torch path.

        The base class refreshes ``_slot_valid`` and calls ``_outputs.reset_step()``
        every step. Both are dead overhead for Warp backends:

        - All Warp backends ``del valid_slots`` — they read ``slot_count[env]``
          directly inside the kernels.
        - Per-step zero of ``buf_error`` / ``buf_activation`` is unnecessary
          because dispatch overwrites every active slot, and inactive slots are
          masked out in compose (``slot < slot_count``) and metrics
          (multiplied by ``active_mask``). Resample still clears outputs for
          envs whose task changes, so no stale data survives a task switch.

        Saves ~24 µs/step at 16k envs locomotion (Torch ``lt`` + 2× ``zero_()``
        + their stream-sync overhead with the captured Warp graph).
        """
        self._dispatch(self._slot_valid)
        self._compose(self._slot_valid)

    def _dispatch(self, valid_slots: torch.Tensor) -> None:
        """State → delta → metric → activation through the selected backend."""
        assert self._backend is not None
        self._backend.dispatch(self, valid_slots)

    def _compose(self, valid_slots: torch.Tensor) -> None:
        """Advance composer state and write terminal reward outputs."""
        assert self._backend is not None
        self._backend.compose(self, valid_slots)
