# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-side visual-material writes for renderers backed by Fabric."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.renderers.base_renderer import VisualMaterialBatch


@wp.kernel(enable_backward=False)
def _write_float(
    values: wp.array(dtype=wp.float32),
    material_offsets: wp.array(dtype=wp.int32),
    env_ids: wp.array(dtype=wp.int32),
    inverse: wp.array(dtype=wp.int32),
    output: wp.fabricarray(dtype=wp.float32),
):
    material, env = wp.tid()
    row = material_offsets[material] + env_ids[env]
    output_row = inverse[row]
    if output_row >= 0:
        output[output_row] = values[row]


@wp.kernel(enable_backward=False)
def _write_float2(
    values: wp.array(dtype=wp.vec2f),
    material_offsets: wp.array(dtype=wp.int32),
    env_ids: wp.array(dtype=wp.int32),
    inverse: wp.array(dtype=wp.int32),
    output: wp.fabricarray(dtype=wp.vec2f),
):
    material, env = wp.tid()
    row = material_offsets[material] + env_ids[env]
    output_row = inverse[row]
    if output_row >= 0:
        output[output_row] = values[row]


@wp.kernel(enable_backward=False)
def _write_float3(
    values: wp.array(dtype=wp.vec3f),
    material_offsets: wp.array(dtype=wp.int32),
    env_ids: wp.array(dtype=wp.int32),
    inverse: wp.array(dtype=wp.int32),
    output: wp.fabricarray(dtype=wp.vec3f),
):
    material, env = wp.tid()
    row = material_offsets[material] + env_ids[env]
    output_row = inverse[row]
    if output_row >= 0:
        output[output_row] = values[row]


_WRITES = {
    (torch.float32, ()): (_write_float, wp.float32),
    (torch.float32, (2,)): (_write_float2, wp.vec2f),
    (torch.float32, (3,)): (_write_float3, wp.vec3f),
}


class FabricVisualMaterialWriter:
    """Compile Fabric shader addresses once and update them from stable device buffers."""

    def __init__(self, bind: Any, batches: tuple[VisualMaterialBatch, ...]):
        compiled = []
        for batch in batches:
            layout = (batch.values.dtype, tuple(batch.values.shape[1:]))
            if layout not in _WRITES:
                raise TypeError(f"Unsupported Fabric visual-material tensor layout: {layout}.")
            compiled.append((batch, *_WRITES[layout]))

        writes = {batch.channel: [] for batch, _kernel, _dtype in compiled}
        for batch, kernel, dtype in compiled:
            values = wp.from_torch(batch.values.detach(), dtype=dtype)
            groups: dict[str, list[int]] = {}
            for row, input_name in enumerate(batch.input_names):
                groups.setdefault(input_name, []).append(row)
            for input_name, rows in groups.items():
                selection, inverse = bind(batch, input_name, tuple(rows))
                attribute_name = f"inputs:{input_name}"
                writes[batch.channel].append((kernel, values, selection, inverse, attribute_name))

        self._writes = {channel: tuple(channel_writes) for channel, channel_writes in writes.items()}
        self._all_offsets = {
            batch.channel: wp.array(range(len(batch.values)), dtype=wp.int32, device=str(batch.values.device))
            for batch, _kernel, _dtype in compiled
        }
        device = next(iter(compiled))[0].values.device
        self._zero_env_id = wp.zeros(1, dtype=wp.int32, device=str(device))

    def __call__(self, material_offsets: dict[str, wp.array] | None = None, env_ids: wp.array | None = None) -> None:
        """Write selected material/environment rows through precompiled Fabric selections."""
        selected = self._all_offsets if material_offsets is None else material_offsets
        env_ids = self._zero_env_id if env_ids is None else env_ids
        for channel, offsets in selected.items():
            for kernel, values, selection, inverse, attribute_name in self._writes[channel]:
                selection.PrepareForReuse()
                wp.launch(
                    kernel,
                    dim=(len(offsets), len(env_ids)),
                    inputs=[values, offsets, env_ids, inverse, wp.fabricarray(selection, attribute_name)],
                    device=values.device,
                )

    def close(self) -> None:
        """Release references to stage-bound Fabric selections."""
        self._all_offsets = {}
        self._zero_env_id = None
        self._writes = {}
