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

Selected when ``MultiTaskCfg.dispatch_backend`` is not ``"reference"``. The
factory in :class:`~.multi_task_command.MultiTaskCommand.__new__` routes
construction to this class automatically; users never reference it directly.
"""

from __future__ import annotations

import torch

from .impl import CommandBackend, build_command_backend, build_command_output_store
from .multi_task_command import MultiTaskCommand

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

    def _dispatch(self, valid_slots: torch.Tensor) -> None:
        """State → delta → metric → activation through the selected backend."""
        assert self._backend is not None
        self._backend.dispatch(self, valid_slots)

    def _compose(self, valid_slots: torch.Tensor) -> None:
        """Advance composer state and write terminal reward outputs."""
        assert self._backend is not None
        self._backend.compose(self, valid_slots)
