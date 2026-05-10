# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read phase for the current mega-kernel backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

import warp as wp

from ... import multi_task_command as _base_module
from ...kernels_torch import BUFFER_KIND
from ...kernels_wp import fill_slab_body_pos_env_local, fill_slab_copy

if TYPE_CHECKING:
    from ...multi_task_command_warp import MultiTaskCommandWarp
    from .bindings import MegaKernelPlan


def fill_unified_buffer_warp(command: MultiTaskCommandWarp, plan: MegaKernelPlan) -> None:
    """Read scene slabs into the unified Warp dispatch buffer."""
    slab_kinds = command.spec.slab_buffer_kinds
    slab_assets = command.spec.slab_asset_names
    slab_offsets = command.spec.slab_offsets_py
    slab_sizes = command.spec.slab_sizes_py
    device_str = str(command.device)
    unified_wp = plan.state.unified
    body_pos_w_kind = int(BUFFER_KIND.BODY_POS_W)
    for slab_id in range(len(slab_kinds)):
        kind = slab_kinds[slab_id]
        asset_name = slab_assets[slab_id]
        offset = slab_offsets[slab_id]
        size = slab_sizes[slab_id]
        raw = _base_module.BUFFER_KIND_READERS[kind](command._env, asset_name)
        raw_per_env = raw.numel() // command.num_envs
        if raw_per_env != size:
            raise RuntimeError(
                f"State kernel output dim mismatch for slab (kind={kind}, asset={asset_name}): "
                f"reader returned {raw_per_env} floats per env, but slab was sized for {size}."
            )
        source = raw.reshape(command.num_envs, size)
        if kind == body_pos_w_kind:
            wp.launch(
                fill_slab_body_pos_env_local,
                dim=(command.num_envs, size),
                inputs=[
                    wp.from_torch(source),
                    wp.from_torch(command._env.scene.env_origins),
                    unified_wp,
                    offset,
                ],
                device=device_str,
            )
        else:
            wp.launch(
                fill_slab_copy,
                dim=(command.num_envs, size),
                inputs=[wp.from_torch(source), unified_wp, offset],
                device=device_str,
            )
