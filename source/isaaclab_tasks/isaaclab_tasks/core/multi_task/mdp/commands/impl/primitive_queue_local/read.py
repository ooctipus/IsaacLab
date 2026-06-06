# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read phase for primitive_queue_local — fills the unified buffer from typed slabs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ..kernels_wp import (
    fill_slab_copy,
    fill_slab_joint_mech_power_abs,
    fill_slab_quat,
    fill_slab_vec3,
    fill_slab_vec3_env_local,
)

if TYPE_CHECKING:
    from ..multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import PrimitiveQueueLocalPlan


def fill_unified_buffer_warp(command: MultiTaskCommandWarp, plan: PrimitiveQueueLocalPlan) -> None:
    """Fill the unified scene buffer from per-kind typed sources."""
    device_str = str(command.device)
    unified_wp = plan.state.unified

    for slab in plan.float_slabs:
        wp.launch(
            fill_slab_copy,
            dim=(command.num_envs, slab.size),
            inputs=[slab.source_wp, unified_wp, slab.offset],
            device=device_str,
        )
    for slab in plan.vec3_slabs:
        wp.launch(
            fill_slab_vec3,
            dim=(command.num_envs, slab.size // 3),
            inputs=[slab.source_wp, unified_wp, slab.offset],
            device=device_str,
        )
    for slab in plan.vec3_env_local_slabs:
        wp.launch(
            fill_slab_vec3_env_local,
            dim=(command.num_envs, slab.size // 3),
            inputs=[slab.source_wp, slab.env_origins_wp, unified_wp, slab.offset],
            device=device_str,
        )
    for slab in plan.quat_slabs:
        wp.launch(
            fill_slab_quat,
            dim=(command.num_envs, slab.size // 4),
            inputs=[slab.source_wp, unified_wp, slab.offset],
            device=device_str,
        )
    for slab in plan.joint_mech_power_slabs:
        wp.launch(
            fill_slab_joint_mech_power_abs,
            dim=(command.num_envs, slab.size),
            inputs=[slab.applied_torque_wp, slab.joint_vel_wp, unified_wp, slab.offset],
            device=device_str,
        )
