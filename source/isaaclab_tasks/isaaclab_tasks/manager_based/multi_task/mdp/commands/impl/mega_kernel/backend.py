# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend object for the current mega-kernel execution layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .bindings import MegaKernelPlan, build_mega_kernel_plan
from .compose import compose_warp
from .execute import dispatch_mega_warp
from .read import fill_unified_buffer_warp
from .rotation import rotate_canonical_slots_to_body_frame_warp

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp


class MegaKernelBackend:
    """Execute ``MultiTaskCommand`` through the mega-kernel ``(env, slot)`` plan."""

    name = "mega_kernel"

    def __init__(self, command: MultiTaskCommandWarp):
        self.plan: MegaKernelPlan = build_mega_kernel_plan(command)

    def on_resample(self, command: MultiTaskCommandWarp, env_ids: torch.Tensor) -> None:
        """No-op: env-slot tensors are wrapped directly and mutate in place."""
        del command, env_ids

    def dispatch(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Run read, execute, and body-frame rotation phases."""
        del valid_slots
        fill_unified_buffer_warp(command, self.plan)
        dispatch_mega_warp(command, self.plan)
        rotate_canonical_slots_to_body_frame_warp(command, self.plan)

    def compose(self, command: MultiTaskCommandWarp, valid_slots: torch.Tensor) -> None:
        """Advance composer state and write reward outputs."""
        del valid_slots
        compose_warp(command, self.plan)
