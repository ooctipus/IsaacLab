# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read phase for the current mega-kernel backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ... import multi_task_command as _base_module
from ...kernels_wp import fill_slab_body_pos_env_local, fill_slab_copy

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import MegaKernelPlan


def fill_unified_buffer_warp(command: MultiTaskCommandWarp, plan: MegaKernelPlan) -> None:
    """Read scene slabs into the unified Warp dispatch buffer.

    Stable scene-backed slabs are launched directly from prebound ``wp.array``
    handles. Slabs whose readers allocate fresh tensors per call (detected at
    plan construction) fall back to a per-step ``wp.from_torch`` rebind.
    """
    device_str = str(command.device)
    unified_wp = plan.state.unified

    for slab in plan.copy_slabs:
        wp.launch(
            fill_slab_copy,
            dim=(command.num_envs, slab.size),
            inputs=[slab.source_wp, unified_wp, slab.offset],
            device=device_str,
        )
    for slab in plan.body_pos_slabs:
        wp.launch(
            fill_slab_body_pos_env_local,
            dim=(command.num_envs, slab.size),
            inputs=[slab.source_wp, slab.env_origins_wp, unified_wp, slab.offset],
            device=device_str,
        )
    for slab in plan.dynamic_slabs:
        raw = _base_module.BUFFER_KIND_READERS[slab.kind](command._env, slab.asset_name)
        source = raw.reshape(command.num_envs, slab.size)
        if slab.is_body_pos:
            wp.launch(
                fill_slab_body_pos_env_local,
                dim=(command.num_envs, slab.size),
                inputs=[wp.from_torch(source), slab.env_origins_wp, unified_wp, slab.offset],
                device=device_str,
            )
        else:
            wp.launch(
                fill_slab_copy,
                dim=(command.num_envs, slab.size),
                inputs=[wp.from_torch(source), unified_wp, slab.offset],
                device=device_str,
            )
