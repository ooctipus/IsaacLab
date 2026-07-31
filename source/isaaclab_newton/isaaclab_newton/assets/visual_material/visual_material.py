# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.assets.visual_material.base_visual_material import (
    BaseVisualMaterial,
    _bucket_writes,
    _notify_renderers,
    _per_env_writes,
    _resolve_channel_specs,
    _validate_per_env_texture_paths,
    _validated_bucket_values,
    _validated_per_env_values,
)
from isaaclab.renderers.base_renderer import VisualMaterialWrite

from isaaclab_newton.physics.newton_manager import NewtonManager


class VisualMaterial(BaseVisualMaterial):
    """Newton visual material: runtime writes go to the Newton model, not the USD stage.

    Initialization still authors the USD material (the importer bakes the stage into the Newton
    model at finalize), but no Newton consumer reads the stage afterwards. Runtime writes
    therefore skip USD authoring and take two paths:

    * ``color`` scatters into ``model.shape_color`` directly — the tiled camera, the Newton
      viewer, and camera-less runs all read model state.
    * Every write (color included) is also broadcast through
      :meth:`~isaaclab.renderers.render_context.RenderContext.notify_visual_material_written`,
      because renderers other than the model's consumers may be attached under Newton physics
      (e.g. OVRTX cameras): each renderer consumes the semantics it represents — the Newton-Warp
      sensor takes ``texture`` (colors reach it through the model), OVRTX takes colors and
      scalars.

    All written values are cached: :meth:`read_channel` returns the last written value, falling
    back to the USD default authored at initialization.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._written: dict[str, Any] = {}
        self._env_written: dict[str, torch.Tensor | list[str]] = {}

    @classmethod
    def _write_bucket_channels(
        cls, materials: Sequence[VisualMaterial], channels: dict[str, torch.Tensor | list[str]]
    ) -> None:
        """Cache bucket values, scatter colors into the Newton model, and notify renderers."""
        writes = []
        for name, values in channels.items():
            specs = _resolve_channel_specs(materials, name)
            if specs[0].semantic == "texture":
                if len(values) != len(materials):
                    raise ValueError(f"Channel '{name}' expects {len(materials)} asset paths; got {len(values)}.")
                for material, path in zip(materials, values, strict=True):
                    material._written[name] = path
            else:
                values = _validated_bucket_values(values, len(materials), name)
                for material, row in zip(materials, values, strict=True):
                    material._written[name] = tuple(row.tolist()) if row.numel() > 1 else float(row)
                if specs[0].semantic == "color":
                    cls._write_shape_colors(tuple(material.material_prim_path for material in materials), values)
            writes += _bucket_writes(materials, specs, values)
        _notify_renderers(writes)

    @classmethod
    def _write_per_env_channels(
        cls,
        materials: Sequence[VisualMaterial],
        channels: dict[str, torch.Tensor | list[str]],
        env_ids: torch.Tensor | Sequence[int] | None,
    ) -> None:
        """Cache per-environment values, scatter colors into the Newton model, and notify renderers.

        Skips USD authoring like the bucket branch. Each entity keeps a persistent per-env cache
        of the last written values so a partial ``env_ids`` fire still issues full-tuple color
        scatters and texture notifies — the downstream memoized plans then stay keyed by one
        stable (num_envs * num_materials) path tuple instead of one plan per env-id subset.
        """
        num_envs = len(materials[0].env_material_paths)
        selected = list(range(num_envs)) if env_ids is None else [int(env_id) for env_id in env_ids]
        selected_rows = torch.tensor(selected, dtype=torch.long)

        writes = []
        for name, values in channels.items():
            specs = _resolve_channel_specs(materials, name)
            if specs[0].semantic == "texture":
                _validate_per_env_texture_paths(values, len(materials), len(selected), name)
                for material, material_paths in zip(materials, values, strict=True):
                    cache = material._env_written.get(name)
                    if cache is None:
                        default = BaseVisualMaterial.read_channel(material, name)
                        cache = [getattr(default, "path", None) or str(default or "")] * num_envs
                        material._env_written[name] = cache
                    for env_id, path in zip(selected, material_paths, strict=True):
                        cache[env_id] = path
                # full-tuple notify: the Newton renderer memoizes its texture-pool scatter plan per
                # notified path tuple, so unchanged envs re-send their cached path (idempotent)
                writes.append(
                    VisualMaterialWrite(
                        material_paths=[path for m in materials for path in m.env_material_paths],
                        shader_paths=[path for m in materials for path in m._env_shader_paths],
                        attr_name=f"inputs:{specs[0].input_name}",
                        semantic="texture",
                        values=[path for m in materials for path in m._env_written[name]],
                    )
                )
                continue

            values = _validated_per_env_values(values, len(materials), len(selected), name)
            for material, material_rows in zip(materials, values, strict=True):
                cache = material._env_written.get(name)
                if cache is None:
                    cache = material._initial_env_values(name, num_envs, values.shape[-1])
                    material._env_written[name] = cache
                cache[selected_rows] = material_rows
            if specs[0].semantic == "color":
                # full-tuple scatter: unchanged envs re-write their cached color (idempotent)
                paths = tuple(path for material in materials for path in material.env_material_paths)
                cls._write_shape_colors(paths, torch.cat([material._env_written[name] for material in materials]))
            writes += _per_env_writes(materials, specs, values, selected)
        _notify_renderers(writes)

    def _initial_env_values(self, channel: str, num_envs: int, num_components: int) -> torch.Tensor:
        """Build the (num_envs, c) cache seed from the init-authored source value of the channel."""
        default = super().read_channel(channel)
        if default is None:
            row = torch.zeros(num_components, dtype=torch.float32)
        elif hasattr(default, "__len__"):
            row = torch.tensor([float(component) for component in default], dtype=torch.float32)
        else:
            row = torch.tensor([float(default)], dtype=torch.float32)
        return row.repeat(num_envs, 1)

    @staticmethod
    def _write_shape_colors(material_paths: tuple[str, ...], colors: torch.Tensor) -> None:
        """Scatter one linear RGB color per material into ``model.shape_color``."""
        NewtonManager.get_visual_material_writer().write_colors(material_paths, colors)

    def read_channel(self, channel: str, env_id: int | None = None) -> Any:
        """Return the last written value, or the USD default authored at initialization.

        Args:
            channel: The channel name.
            env_id: For per-environment materials, the environment whose value to read. None reads
                the source-environment prim (the only prim of a bucket material).
        """
        if env_id is not None and self.is_per_env:
            cache = self._env_written.get(channel)
            if isinstance(cache, list):
                return cache[env_id]
            if cache is not None:
                row = cache[env_id]
                return tuple(row.tolist()) if row.numel() > 1 else float(row)
            # before the first write every clone holds the init-authored value; read it from the
            # source prim (kitless Newton stages have no USD clones to read)
            return super().read_channel(channel)
        if channel in self._written:
            return self._written[channel]
        return super().read_channel(channel, env_id)
