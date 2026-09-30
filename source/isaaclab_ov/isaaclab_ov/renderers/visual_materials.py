# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Device-only visual-material writes for OVRTX-owned scenes."""

from __future__ import annotations

import itertools
import weakref
from typing import TYPE_CHECKING, Any

import torch
import warp as wp
from ovrtx import DataAccess

if TYPE_CHECKING:
    from isaaclab.renderers.base_renderer import VisualMaterialBatch

    from isaaclab_ov.cloner import OvReplicateContext


class OVRTXVisualMaterialWriter:
    """Compile OVRTX material addresses once and publish dirty device buffers."""

    def __init__(self, context: OvReplicateContext, batches: tuple[VisualMaterialBatch, ...]):
        self._context_ref = weakref.ref(context)
        self._buffers: dict[str, torch.Tensor] = {}
        self._dirty_channels: set[str] = set()
        self._operations: tuple[Any, ...] = ()
        self._addresses: list[tuple[str, Any, slice]] = []

        groups = []
        device = None
        for batch in batches:
            values = batch.values.detach()
            if values.dtype == torch.float32 and values.ndim == 1:
                dtype, shape = "float32", None
            elif values.dtype == torch.float32 and values.ndim == 2 and values.shape[1] in (2, 3):
                dtype, shape = "float32", (values.shape[1],)
            else:
                raise TypeError(
                    f"OVRTX visual-material channel {batch.channel!r} requires float, float2, or float3; "
                    f"got dtype={values.dtype}, shape={tuple(values.shape)}."
                )
            if not values.is_cuda or (device is not None and values.device != device):
                raise RuntimeError("OVRTX visual-material attributes must reside on one CUDA device.")
            device = values.device
            self._buffers[batch.channel] = values
            start = 0
            for input_name, input_group in itertools.groupby(batch.input_names):
                end = start + sum(1 for _ in input_group)
                rows = slice(start, end)
                groups.append(
                    (batch.channel, f"inputs:{input_name}", list(batch.shader_paths[rows]), rows, dtype, shape)
                )
                start = end
        self._event = wp.Event(device=str(device))
        scene = context.scene
        try:
            for channel, attribute_name, shader_paths, rows, dtype, shape in groups:
                handle = scene.bind(shader_paths, attribute_name, dtype=dtype, shape=shape)
                self._addresses.append((channel, handle, rows))
        except Exception:
            self._release_backend_addresses(context)
            raise

    def __call__(self, material_offsets: dict[str, Any] | None = None, env_ids: Any | None = None) -> None:
        """Mark channels dirty; OVRTX currently copies each dirty channel's full device buffer."""
        del env_ids
        selected = self._buffers if material_offsets is None else material_offsets
        self._dirty_channels.update(selected)
        wp.record_event(self._event)

    def publish(self) -> None:
        """Submit dirty buffers to OVRTX after ordering their producer streams."""
        channels = self._dirty_channels
        if not channels:
            return
        if self._context_ref() is None:
            return
        operations = []
        try:
            for channel, handle, rows in self._addresses:
                if channel not in channels:
                    continue
                operation = handle.binding.write_async(
                    self._buffers[channel][rows],
                    data_access=DataAccess.ASYNC,
                    cuda_event=self._event.cuda_event,
                )
                operations.append(operation)
        finally:
            self._operations = tuple(operations)
        channels.clear()

    def drain(self) -> None:
        """Complete submitted writes before their scene buffers may change."""
        operations, self._operations = self._operations, ()
        for operation in operations:
            operation.wait()

    def _release_backend_addresses(self, context: OvReplicateContext) -> None:
        if not self._addresses:
            return
        scene = context.scene
        for _channel, handle, _rows in self._addresses:
            scene.release(handle)
        self._addresses.clear()

    def close(self) -> None:
        """Drain writes and release every compiled backend address."""
        try:
            self.drain()
        finally:
            context = self._context_ref()
            if context is not None:
                self._release_backend_addresses(context)
                if context._visual_material_writer is self:
                    context._visual_material_writer = None
            self._dirty_channels.clear()
            self._buffers.clear()
