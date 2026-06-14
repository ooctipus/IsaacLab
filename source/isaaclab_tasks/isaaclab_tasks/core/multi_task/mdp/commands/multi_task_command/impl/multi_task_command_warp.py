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

This wrapper owns the torch → Warp conversion for the command's mutable
state (spec, env-slot tables, unified buffer, targets, composer state, output
buffers, episode-length state). Each backend consumes the shared Warp views
exposed here through ``command.env_slots_wp`` / ``command.spec_wp`` /
``command.state_wp`` / ``command.composer_state_wp`` / ``command.outputs_wp``
plus ``command.episode_length_buf_wp`` / ``command.effective_max_episode_length_wp`` —
the backend directories stay pure Warp (no ``wp.from_torch`` calls of their own).
"""

from __future__ import annotations

import torch
import warp as wp

from ..multi_task_command import MultiTaskCommand
from . import CommandBackend, build_command_backend
from .kernels_wp import ComposerState, EnvSlots, Outputs, StateAccess, SubtaskSpec

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
        self._build_shared_wp_views()
        self._backend = build_command_backend(self, cfg.dispatch_backend)

    # ------------------------------------------------------------------------
    # Shared Warp views over the base class's torch state.
    #
    # Built once at __init__. Backends consume these directly; no backend
    # makes its own ``wp.from_torch`` call against the command's state.
    # ------------------------------------------------------------------------

    def _build_shared_wp_views(self) -> None:
        s = self.spec

        self.env_slots_wp = EnvSlots()
        self.env_slots_wp.subtask_ids = wp.from_torch(self._env_subtask_ids)
        self.env_slots_wp.slot_count = wp.from_torch(self._env_slot_count)
        self.env_slots_wp.slot_offsets = wp.from_torch(self._env_slot_offsets)

        self.spec_wp = SubtaskSpec()
        self.spec_wp.state_kernel_id = wp.from_torch(s.state_kernel_id.int())
        self.spec_wp.metric_kernel_id = wp.from_torch(s.metric_kernel_id.int())
        self.spec_wp.activation_kernel_id = wp.from_torch(s.activation_kernel_id.int())
        self.spec_wp.activation_kernel_param = wp.from_torch(s.activation_kernel_param)
        self.spec_wp.state_stride = wp.from_torch(s.state_stride.int())
        self.spec_wp.canonical_offset = wp.from_torch(s.canonical_offset.int())
        self.spec_wp.is_instant_flag = wp.from_torch(s.is_instant.int())
        self.spec_wp.is_tracking_flag = wp.from_torch(s.is_tracking.int())
        self.spec_wp.gather_offset = wp.from_torch(s.subtask_gather_offset.int())
        self.spec_wp.gather_count = wp.from_torch(s.subtask_gather_count.int())
        self.spec_wp.gather_indices_flat = wp.from_torch(s.gather_indices_flat.int())

        self.state_wp = StateAccess()
        self.state_wp.unified = wp.from_torch(self._unified_buffer)
        self.state_wp.targets_flat = wp.from_torch(self._targets_flat)

        self.composer_state_wp = ComposerState()
        self.composer_state_wp.sum_activation = wp.from_torch(self._sum_activation)
        self.composer_state_wp.transit_steps = wp.from_torch(self._transit_steps)
        self.composer_state_wp.instant_achieved = wp.from_torch(self._instant_achieved)

        self.outputs_wp = Outputs()
        self.outputs_wp.buf_error = wp.from_torch(self._buf_error)
        self.outputs_wp.buf_activation = wp.from_torch(self._buf_activation)
        self.outputs_wp.command_reach = wp.from_torch(self._command_reach)
        self.outputs_wp.command_track = wp.from_torch(self._command_track)
        self.outputs_wp.task_reward = wp.from_torch(self._task_reward)
        self.outputs_wp.task_done_success = wp.from_torch(self._task_done_success)
        self.outputs_wp.progress = wp.from_torch(self._progress)

        self.episode_length_buf_wp = wp.from_torch(self._env.episode_length_buf)
        self.effective_max_episode_length_wp = wp.from_torch(self._effective_max_episode_length)

    # ------------------------------------------------------------------------
    # Lifecycle hooks.
    # ------------------------------------------------------------------------

    def _on_resample_command(self, env_ids: torch.Tensor) -> None:
        """Refresh backend-owned execution plans after task assignment changes."""
        if self._backend is not None:
            self._backend.on_resample(self, env_ids)

    def _update_command(self) -> None:
        """Per-step Warp update — skip the Torch overhead the base class adds for the Torch path.

        The base class refreshes ``_slot_valid`` and zeros ``_buf_error``/``_buf_activation``
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
        del valid_slots  # Warp backends read ``slot_count[env]`` directly inside kernels.
        assert self._backend is not None
        self._backend.dispatch(self)

    def _compose(self, valid_slots: torch.Tensor) -> None:
        """Advance composer state and write terminal reward outputs."""
        del valid_slots
        assert self._backend is not None
        self._backend.compose(self)
