# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend object for the current mega-kernel execution layout.

Supports two slot-ordering modes through the same execution pipeline:

* ``"natural"`` — slots stay in their authored order (default; the canonical
  ``mega_kernel`` backend).
* ``"schedule"`` — on resample, slot tables are sorted by fused-schedule id so
  each warp sees a coherent state-kernel region inside ``dispatch_mega``.
  Selected via ``dispatch_backend="schedule_ordered_mega"``.

Same plan, same kernels, same launches — only the slot ordering established
at resample time differs.

Pure Warp — no ``import torch``. The per-resample sort calls tensor methods
on Warp-typed views of the command's slot tables (no ``torch.X`` symbols).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import warp as wp

from ..compose_select import use_parallel_compose
from ..kernels_wp import dispatch_compose_fused
from ..schedules import NUM_SCHEDULES, build_subtask_schedule_ids
from .bindings import MegaKernelPlan, build_mega_kernel_plan
from .compose import compose_warp
from .execute import dispatch_mega_warp
from .read import fill_unified_buffer_warp
from .rotation import rotate_canonical_slots_to_body_frame_warp

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp


SlotOrder = Literal["natural", "schedule"]


class MegaKernelBackend:
    """Execute ``MultiTaskCommand`` through the mega-kernel ``(env, slot)`` plan."""

    name = "mega_kernel"

    def __init__(self, command: MultiTaskCommandWarp, slot_order: SlotOrder = "natural"):
        self.plan: MegaKernelPlan = build_mega_kernel_plan(command)
        self._dispatch_graph: wp.Graph | None = None
        self._slot_order: SlotOrder = slot_order
        # Cache the subtask→schedule-id lookup table once at plan build for
        # ``slot_order == "schedule"``. Stored as a Warp-owned array; refresh
        # reads it through a ``wp.to_torch`` view for the indexing math.
        self._subtask_schedule_ids_wp: wp.array | None = None
        if slot_order == "schedule":
            self._subtask_schedule_ids_wp = build_subtask_schedule_ids(
                command.spec.state_kernel_id,
                backend_name="mega_kernel(slot_order=schedule)",
            )
        # Fuse dispatch + compose into one block-per-env kernel when k_max is
        # large enough to fill warps. Same threshold as the parallel compose
        # path (warp size). Saves the buf_activation global-memory roundtrip
        # between dispatch and compose.
        self._use_fused = use_parallel_compose(command.k_max)
        # ``slot_order="schedule"`` needs the initial sort applied to every env.
        if slot_order == "schedule":
            self._resort_slots(command, env_ids=None)

    def on_resample(self, command: MultiTaskCommandWarp, env_ids) -> None:
        """Refresh slot ordering when the mode requires it."""
        if self._slot_order == "schedule":
            self._resort_slots(command, env_ids)

    def _resort_slots(self, command: MultiTaskCommandWarp, env_ids) -> None:
        """Sort each resampled env's slot tables by fused-schedule id (stable).

        Active slots sort first by schedule; inactive (``slot >= slot_count``)
        slots get ``NUM_SCHEDULES`` so they land at the end. This gives
        ``dispatch_mega`` warp-coherent state-kernel regions.

        ``env_ids=None`` selects all envs (initial sort at backend
        construction); otherwise ``env_ids`` is a torch index tensor of the
        envs being resampled this step.
        """
        if env_ids is not None and env_ids.numel() == 0:
            return
        assert self._subtask_schedule_ids_wp is not None
        schedule_ids_torch = wp.to_torch(self._subtask_schedule_ids_wp)

        if env_ids is None:
            n_rows = command.num_envs
            subtask_ids_block = command._env_subtask_ids
            slot_count_block = command._env_slot_count
            offsets_block = command._env_slot_offsets
            strides_block = command._env_slot_strides
        else:
            n_rows = env_ids.numel()
            subtask_ids_block = command._env_subtask_ids[env_ids]
            slot_count_block = command._env_slot_count[env_ids]
            offsets_block = command._env_slot_offsets[env_ids]
            strides_block = command._env_slot_strides[env_ids]

        slot_ids = command._slot_arange.expand(n_rows, -1)
        active = slot_ids < slot_count_block.unsqueeze(1)
        clamped = subtask_ids_block.long().clamp_min(0)
        schedule_ids = schedule_ids_torch[clamped]
        fallback = schedule_ids.new_full(schedule_ids.shape, NUM_SCHEDULES)
        schedule_ids = schedule_ids.where(active, fallback)
        slot_order = schedule_ids.argsort(dim=1, stable=True)

        sorted_subtask = subtask_ids_block.gather(1, slot_order)
        sorted_offsets = offsets_block.gather(1, slot_order)
        sorted_strides = strides_block.gather(1, slot_order)

        if env_ids is None:
            command._env_subtask_ids[:] = sorted_subtask
            command._env_slot_offsets[:] = sorted_offsets
            command._env_slot_strides[:] = sorted_strides
        else:
            command._env_subtask_ids[env_ids] = sorted_subtask
            command._env_slot_offsets[env_ids] = sorted_offsets
            command._env_slot_strides[env_ids] = sorted_strides

    def dispatch(self, command: MultiTaskCommandWarp) -> None:
        """Run the full per-step pipeline (read + dispatch + rotate + compose) through a captured graph.

        The captured graph includes ``compose`` so the public ``compose()`` hook
        becomes a no-op — saves one launch + one host-side stream synchronization
        per step relative to launching compose separately.
        """
        if wp.get_device(str(command.device)).is_capturing:
            self._dispatch_uncaptured(command)
            return
        if self._dispatch_graph is None:
            self._dispatch_uncaptured(command)
            with wp.ScopedCapture(device=str(command.device)) as capture:
                self._dispatch_uncaptured(command)
            self._dispatch_graph = capture.graph
            return
        wp.capture_launch(self._dispatch_graph)

    def _dispatch_uncaptured(self, command: MultiTaskCommandWarp) -> None:
        """Launch the full per-step pipeline eagerly; used for warmup and graph capture."""
        fill_unified_buffer_warp(command, self.plan)
        if self._use_fused:
            wp.launch_tiled(
                dispatch_compose_fused,
                dim=[command.num_envs],
                inputs=[
                    self.plan.env_slots,
                    self.plan.spec,
                    self.plan.state,
                    self.plan.outputs,
                    self.plan.composer_state,
                    self.plan.episode_length_buf_wp,
                    self.plan.effective_max_episode_length_wp,
                    0.5,
                    float(command.cfg.quality_easing),
                    self.plan.inline_rotation_quat_wp,
                    self.plan.subtask_is_rotatable_wp,
                    self.plan.use_inline_rotation,
                ],
                block_dim=max(command.k_max, 32),
                device=str(command.device),
            )
            # Skip the standalone rotate launch when inline rotation handled it.
            if not self.plan.use_inline_rotation:
                rotate_canonical_slots_to_body_frame_warp(command, self.plan)
        else:
            dispatch_mega_warp(command, self.plan)
            rotate_canonical_slots_to_body_frame_warp(command, self.plan)
            compose_warp(command, self.plan)

    def compose(self, command: MultiTaskCommandWarp) -> None:
        """No-op — compose was captured as part of the dispatch graph."""
        del command
