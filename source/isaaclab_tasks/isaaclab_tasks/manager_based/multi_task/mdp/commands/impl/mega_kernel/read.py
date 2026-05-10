# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read phase for the current mega-kernel backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ... import multi_task_command as _base_module
from ...kernels_wp import fill_slab_body_pos_env_local, fill_slab_copy, fill_slabs_combined_8

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import MegaKernelPlan


def fill_unified_buffer_warp(command: MultiTaskCommandWarp, plan: MegaKernelPlan) -> None:
    """Read scene slabs into the unified Warp dispatch buffer.

    Stable copy slabs are launched as one fused ``fill_slabs_combined_8``
    kernel when their count fits the fixed-arity slot limit; otherwise each
    runs as a separate ``fill_slab_copy``. body_pos and dynamic-reader slabs
    keep their own launches.
    """
    device_str = str(command.device)
    unified_wp = plan.state.unified

    # Combined-copy launch is opt-in; not all plan classes carry the metadata.
    combined_num_slabs = getattr(plan, "combined_slab_num_slabs", 0)
    if combined_num_slabs > 0:
        s = plan.combined_slab_sources_wp
        wp.launch(
            fill_slabs_combined_8,
            dim=(command.num_envs, plan.combined_slab_total_size),
            inputs=[
                s[0],
                s[1],
                s[2],
                s[3],
                s[4],
                s[5],
                s[6],
                s[7],
                plan.combined_slab_cumsizes_wp,
                plan.combined_slab_offsets_wp,
                unified_wp,
                combined_num_slabs,
            ],
            device=device_str,
        )
    else:
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
