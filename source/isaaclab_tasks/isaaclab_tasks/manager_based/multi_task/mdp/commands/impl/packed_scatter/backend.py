# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend object for a packed fused-pipeline queue with legacy output scatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..mega_kernel.compose import compose_warp
from ..mega_kernel.read import fill_unified_buffer_warp
from ..mega_kernel.rotation import rotate_canonical_slots_to_body_frame_warp
from .bindings import PackedScatterPlan, build_packed_scatter_plan, refresh_packed_scatter_plan
from .execute import dispatch_packed_scatter_warp

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp


class PackedScatterBackend:
    """Execute through a packed fused-pipeline queue and scatter to legacy outputs."""

    name = "packed_scatter"

    def __init__(self, command: MultiTaskCommandWarp):
        self.plan: PackedScatterPlan = build_packed_scatter_plan(command)

    def on_resample(self, command: MultiTaskCommandWarp, env_ids: torch.Tensor) -> None:
        """Refresh packed queues after task assignment changes."""
        del env_ids
        refresh_packed_scatter_plan(command, self.plan)

    def dispatch(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Run read, packed execution, and body-frame rotation phases."""
        del valid_slots
        fill_unified_buffer_warp(command, self.plan)
        dispatch_packed_scatter_warp(command, self.plan)
        rotate_canonical_slots_to_body_frame_warp(command, self.plan)

    def compose(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Advance composer state and write reward outputs."""
        del valid_slots
        compose_warp(command, self.plan)
