# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Body-frame command rotation for the primitive_queue_local backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..kernels_wp import rotate_canonical_vec3_pair

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import PrimitiveQueueLocalPlan


def rotate_canonical_slots_to_body_frame_warp(command: MultiTaskCommandWarp, plan: PrimitiveQueueLocalPlan) -> None:
    """Rotate policy-facing vec3 command slots in Warp."""
    device_str = str(command.device)
    for binding in plan.rotations:
        wp.launch(
            rotate_canonical_vec3_pair,
            dim=(command.num_envs, binding.num_offsets),
            inputs=[
                binding.root_quat_w_wp,
                plan.outputs.command_reach,
                binding.reach_offsets_wp,
                binding.num_reach_offsets,
                plan.outputs.command_track,
                binding.track_offsets_wp,
            ],
            device=device_str,
        )
