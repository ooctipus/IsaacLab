# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime-writable visual material asset."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import warp as wp

from pxr import Sdf, UsdShade

from isaaclab import cloner
from isaaclab.physics import PhysicsEvent
from isaaclab.renderers.base_renderer import VisualMaterialBatch
from isaaclab.sim import SimulationContext

from .visual_material_cfg import VisualMaterialCfg

_PREVIEW_CHANNELS = {
    "color": ("diffuseColor", (0.18, 0.18, 0.18)),
    "roughness": ("roughness", 0.5),
    "metallic": ("metallic", 0.0),
    "emissive_color": ("emissiveColor", (0.0, 0.0, 0.0)),
    "opacity": ("opacity", 1.0),
}
_PBR_CHANNELS = {
    "color": ("diffuse_color_constant", (0.18, 0.18, 0.18)),
    "roughness": ("reflection_roughness_constant", 0.5),
    "metallic": ("metallic_constant", 0.0),
    "specular": ("specular_level", 0.5),
    "emissive_color": ("emissive_color", (0.0, 0.0, 0.0)),
    "emissive_intensity": ("emissive_intensity", 0.0),
    "opacity": ("opacity_constant", 1.0),
    "uv_scale": ("texture_scale", (1.0, 1.0)),
    "uv_offset": ("texture_translate", (0.0, 0.0)),
    "uv_rotate": ("texture_rotate", 0.0),
}
_GLASS_CHANNELS = {
    "color": ("glass_color", (1.0, 1.0, 1.0)),
    "roughness": ("frosting_roughness", 0.0),
    "ior": ("glass_ior", 1.491),
}


class VisualMaterial:
    """A plan-owned runtime-writable visual material."""

    cfg: VisualMaterialCfg

    def __init__(self, cfg: VisualMaterialCfg):
        cfg.validate()
        sim = SimulationContext.instance()
        if sim is None:
            raise RuntimeError("VisualMaterial requires an active SimulationContext.")
        plan = sim.get_clone_plan()
        if plan is None or plan.is_complete:
            raise RuntimeError("VisualMaterial must be constructed from cfg before the clone plan completes.")
        self.cfg = cfg.copy()
        self.cfg.prim_path = cloner.expand_env_regex_ns(self.cfg.prim_path)
        source_paths = cloner.query.cfg_source_paths(plan, cfg)
        if len(source_paths) != 1 or source_paths[0] is None:
            raise RuntimeError(f"Visual material at {self.cfg.prim_path!r} requires one populated plan source.")
        self._source_material_path = source_paths[0]
        rows = tuple(cloner.query.iter_sources(plan, self.cfg.prim_path))
        destination = next(destination for _, destination, source, _ in rows if source == source_paths[0])
        self._is_per_env = "{}" in destination

        shader = UsdShade.Shader(self.cfg.spawn.func(self._source_material_path, self.cfg.spawn))
        if not shader:
            raise TypeError("VisualMaterial spawners must return the authored UsdShade.Shader prim.")
        self._source_shader_path = str(shader.GetPrim().GetPath())
        shader_suffix = cloner.path.relative_to(self._source_shader_path, self._source_material_path)
        if shader_suffix is None:
            raise ValueError(
                f"Visual material shader {self._source_shader_path!r} is outside {self._source_material_path!r}."
            )

        self._registry = sim.get_or_create_backend(_VisualMaterialRegistry, sim)
        channel_specs = _channel_specs(shader)

        self._input_names: dict[str, str] = {}
        initial_values: dict[str, torch.Tensor] = {}
        for channel in self.cfg.channels:
            if channel not in channel_specs:
                raise ValueError(
                    f"Material {self._source_material_path!r} does not support channel {channel!r}; "
                    f"available channels are {tuple(channel_specs)}."
                )
            input_name, default = channel_specs[channel]
            value_type = (
                Sdf.ValueTypeNames.Float
                if isinstance(default, float)
                else {
                    2: Sdf.ValueTypeNames.Float2,
                    3: Sdf.ValueTypeNames.Color3f,
                }[len(default)]
            )
            shader_input = shader.CreateInput(input_name, value_type)
            if shader_input.Get() is None:
                shader_input.Set(default)
            shader.GetPrim().CreateAttribute(
                f"isaaclab:visualMaterial:{channel}:{input_name}", Sdf.ValueTypeNames.UInt, custom=True
            ).Set(0)
            self._input_names[channel] = input_name
            initial_values[channel] = torch.as_tensor(shader_input.Get(), dtype=torch.float32)

        if self._is_per_env:
            assert plan.env_ids is not None
            columns = {int(env_id): column for column, env_id in enumerate(plan.env_ids)}
            material_paths = [""] * plan.env_ids.size
            for source_root, destination, source_path, env_ids in rows:
                for env_id in env_ids:
                    material_paths[columns[env_id]] = cloner.path.rebase(
                        source_path, source_root, destination.format(env_id)
                    )
            if not all(material_paths):
                raise ValueError(
                    f"Per-environment material {self._source_material_path!r} must populate every environment."
                )
            self._material_paths = tuple(material_paths)
            self._shader_paths = tuple(path + shader_suffix for path in self._material_paths)
        else:
            self._material_paths = (self._source_material_path,)
            self._shader_paths = (self._source_shader_path,)
        self._values = {
            channel: value.to(sim.device).expand(len(self._material_paths), *value.shape).clone()
            for channel, value in initial_values.items()
        }
        self._offsets: dict[str, int] = {}
        self._registry.register(self)

    @property
    def channels(self) -> tuple[str, ...]:
        """Runtime-writable channel names."""
        return tuple(self._input_names)

    @property
    def is_per_env(self) -> bool:
        """Whether this material owns one clone per environment."""
        return self._is_per_env

    @property
    def num_instances(self) -> int:
        return len(self._material_paths)

    @property
    def data(self) -> dict[str, torch.Tensor]:
        return self._values

    @staticmethod
    def write_channels(
        materials: Sequence[VisualMaterial],
        channels: dict[str, torch.Tensor],
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Write aligned numeric channels to bucket or selected environment rows."""
        if materials:
            materials[0]._registry.write(materials, channels, env_ids)


@wp.kernel(enable_backward=False)
def _write_material(
    values: wp.array(dtype=Any, ndim=2),
    offsets: wp.array(dtype=wp.int32),
    env_ids: wp.array(dtype=wp.int32),
    output: wp.array(dtype=Any),
):
    material, env = wp.tid()
    output[offsets[material] + env_ids[env]] = values[material, env]


_MATERIAL_WRITES = {(): wp.float32, (2,): wp.vec2f, (3,): wp.vec3f}


def _channel_specs(shader: UsdShade.Shader) -> dict[str, tuple[str, float | tuple[float, ...]]]:
    if shader.GetShaderId() == "UsdPreviewSurface":
        return dict(_PREVIEW_CHANNELS)
    identifier = shader.GetSourceAssetSubIdentifier("mdl")
    if identifier and identifier.startswith("OmniGlass"):
        return dict(_GLASS_CHANNELS)
    if identifier and identifier.startswith("OmniPBR"):
        channels = dict(_PBR_CHANNELS)
        texture = shader.GetInput("diffuse_texture")
        if texture and texture.Get():
            channels["color"] = ("diffuse_tint", (1.0, 1.0, 1.0))
        return channels
    raise TypeError("VisualMaterial requires PreviewSurfaceCfg, PbrMdlCfg, or GlassMdlCfg.")


class _VisualMaterialRegistry:
    """Compose plan-owned material rows for the renderer set in one simulation."""

    def __init__(self, sim: SimulationContext):
        self._sim = sim
        self._materials: list[VisualMaterial] = []
        self._batches_by_channel: dict[str, VisualMaterialBatch] = {}
        self._batch_views: dict[str, wp.array] = {}
        self._writers: tuple[Any, ...] = ()
        self._selections: dict[tuple[str, tuple[int, ...]], tuple[torch.Tensor, wp.array]] = {}
        self._env_ids: dict[tuple[torch.device, int], tuple[torch.Tensor, wp.array]] = {}
        self._initialized = False
        sim._register_physics_callback(
            self._initialize, PhysicsEvent.PHYSICS_READY, order=40, name="initialize_visual_materials"
        )
        sim._register_physics_callback(self._close, PhysicsEvent.STOP, order=0, name="close_visual_materials")

    def register(self, material: VisualMaterial) -> None:
        """Register one material constructed from the active plan."""
        if any(registered is material for registered in self._materials):
            return
        if self._initialized:
            raise RuntimeError("Visual materials must initialize before rendering consumers.")
        self._materials.append(material)

    def write(
        self,
        materials: Sequence[VisualMaterial],
        channels: dict[str, torch.Tensor],
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Update selected rows and dispatch the compiled renderer writers."""
        if not materials or not channels:
            return
        if not self._initialized:
            raise RuntimeError("Visual materials can only be written after simulation reset.")
        per_env = materials[0].is_per_env
        if not per_env and env_ids is not None:
            raise ValueError("env_ids is only valid for per-environment materials.")

        device = next(iter(self._batches_by_channel.values())).values.device
        count = materials[0].num_instances if per_env else 1
        if env_ids is None:
            key = (device, count)
            selected = self._env_ids.get(key)
            if selected is None:
                env_tensor = torch.arange(count, dtype=torch.int32, device=device)
                selected = (env_tensor, wp.from_torch(env_tensor, dtype=wp.int32))
                self._env_ids[key] = selected
        else:
            env_tensor = env_ids.to(device=device, dtype=torch.int32)
            selected = (env_tensor, wp.from_torch(env_tensor, dtype=wp.int32))

        stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if device.type == "cuda" else None
        with wp.ScopedStream(stream, sync_enter=False):
            material_offsets = {}
            material_key = tuple(id(material) for material in materials)
            for channel, values in channels.items():
                batch = self._batches_by_channel[channel]
                key = (channel, material_key)
                offsets = self._selections.get(key)
                if offsets is None:
                    offset_tensor = torch.tensor(
                        [material._offsets[channel] for material in materials],
                        dtype=torch.int32,
                        device=batch.values.device,
                    )
                    offsets = (offset_tensor, wp.from_torch(offset_tensor, dtype=wp.int32))
                    self._selections[key] = offsets
                trailing = tuple(batch.values.shape[1:])
                expected = (len(materials), len(selected[0]), *trailing)
                values = values.detach().to(device=batch.values.device, dtype=torch.float32)
                if not per_env:
                    values = values.unsqueeze(1)
                if tuple(values.shape) != expected:
                    raise ValueError(
                        f"Channel {channel!r} values must have shape {expected}; got {tuple(values.shape)}."
                    )
                wp.launch(
                    _write_material,
                    dim=(len(materials), len(selected[0])),
                    inputs=[
                        wp.from_torch(values, dtype=_MATERIAL_WRITES[trailing]),
                        offsets[1],
                        selected[1],
                        self._batch_views[channel],
                    ],
                    device=str(batch.values.device),
                )
                material_offsets[channel] = offsets[1]
            for writer in self._writers:
                writer(material_offsets, selected[1])

    def _initialize(self, _payload: Any = None) -> None:
        """Compose flat buffers after assets and render consumers initialize."""
        self._close()
        batches = []
        channels = {channel for material in self._materials for channel in material.channels}
        for channel in sorted(channels):
            rows = sorted(
                (
                    (
                        material,
                        material._material_paths,
                        material._shader_paths,
                        material._input_names[channel],
                        material._values[channel],
                    )
                    for material in self._materials
                    if channel in material.channels
                ),
                key=lambda row: row[3],
            )
            values = torch.cat([row[4] for row in rows])
            batch = VisualMaterialBatch(
                channel,
                tuple(path for row in rows for path in row[1]),
                tuple(path for row in rows for path in row[2]),
                tuple(row[3] for row in rows for _ in row[1]),
                values,
            )
            batches.append(batch)
            offset = 0
            for material, paths, _shader_paths, _input_name, _values in rows:
                end = offset + len(paths)
                material._values[channel] = values[offset:end]
                material._offsets[channel] = offset
                offset = end

        batches = tuple(batches)
        self._batches_by_channel = {batch.channel: batch for batch in batches}
        self._batch_views = {
            batch.channel: wp.from_torch(batch.values, dtype=_MATERIAL_WRITES[tuple(batch.values.shape[1:])])
            for batch in batches
        }
        factories = []
        for consumer in (*self._sim.visualizers, *self._sim._renderer_entries):
            factory = consumer.visual_material_writer
            if factory is not None and factory not in factories:
                factories.append(factory)
        writers = []
        try:
            if batches:
                device = batches[0].values.device
                stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if device.type == "cuda" else None
                with wp.ScopedStream(stream, sync_enter=False):
                    writers = [factory(batches) for factory in factories]
                    for writer in writers:
                        writer()
        except Exception:
            for writer in writers:
                writer.close()
            raise
        self._writers = tuple(writers)
        self._initialized = True

    def _close(self, _payload: Any = None) -> None:
        """Release renderer writers before their native resources stop."""
        errors = self._close_writers()
        self._batches_by_channel.clear()
        self._batch_views.clear()
        self._selections.clear()
        self._env_ids.clear()
        self._initialized = False
        if errors:
            raise RuntimeError(f"{len(errors)} visual-material writer(s) failed to close.") from errors[0]

    def _close_writers(self) -> list[Exception]:
        writers, self._writers = self._writers, ()
        errors: list[Exception] = []
        for writer in writers:
            try:
                writer.close()
            except Exception as exc:  # noqa: BLE001 - close every writer before reporting failure
                errors.append(exc)
        return errors
