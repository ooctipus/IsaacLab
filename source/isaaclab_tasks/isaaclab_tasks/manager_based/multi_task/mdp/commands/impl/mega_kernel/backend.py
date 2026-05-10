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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
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
        # ``slot_order == "schedule"``. Cheap CPU tensor; reused on every resample.
        self._subtask_schedule_ids_i32: torch.Tensor | None = None
        if slot_order == "schedule":
            self._subtask_schedule_ids_i32 = build_subtask_schedule_ids(
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
            self._resort_slots(command, torch.arange(command.num_envs, device=command.device, dtype=torch.long))

    def on_resample(self, command: MultiTaskCommandWarp, env_ids: torch.Tensor) -> None:
        """Refresh slot ordering when the mode requires it."""
        if self._slot_order == "schedule":
            self._resort_slots(command, env_ids)

    def _resort_slots(self, command: MultiTaskCommandWarp, env_ids: torch.Tensor) -> None:
        """Sort each resampled env's slot tables by fused-schedule id (stable).

        Active slots sort first by schedule; inactive (``slot >= slot_count``)
        slots get ``NUM_SCHEDULES`` so they land at the end. This gives
        ``dispatch_mega`` warp-coherent state-kernel regions.
        """
        if env_ids.numel() == 0:
            return
        assert self._subtask_schedule_ids_i32 is not None
        slot_ids = torch.arange(command.k_max, device=command.device, dtype=torch.long).unsqueeze(0)
        slot_ids = slot_ids.expand(env_ids.numel(), -1)
        active = slot_ids < command._env_slot_count[env_ids].long().unsqueeze(1)
        subtask_ids = command._env_subtask_ids[env_ids].long().clamp_min(0)
        schedule_ids = self._subtask_schedule_ids_i32[subtask_ids]
        schedule_ids = torch.where(active, schedule_ids, torch.full_like(schedule_ids, NUM_SCHEDULES))
        slot_order = torch.argsort(schedule_ids, dim=1, stable=True)
        command._env_subtask_ids[env_ids] = torch.gather(command._env_subtask_ids[env_ids], 1, slot_order)
        command._env_slot_offsets[env_ids] = torch.gather(command._env_slot_offsets[env_ids], 1, slot_order)
        command._env_slot_strides[env_ids] = torch.gather(command._env_slot_strides[env_ids], 1, slot_order)

    def dispatch(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Run the full per-step pipeline (read + dispatch + rotate + compose) through a captured graph.

        The captured graph includes ``compose`` so the public ``compose()`` hook
        becomes a no-op — saves one launch + one host-side stream synchronization
        per step relative to launching compose separately.
        """
        del valid_slots
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

    def compose(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """No-op — compose was captured as part of the dispatch graph."""
        del command, valid_slots
