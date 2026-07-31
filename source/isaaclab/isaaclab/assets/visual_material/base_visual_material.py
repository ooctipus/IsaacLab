# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene-level visual material entity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from pxr import Gf, Sdf, UsdShade

from isaaclab.cloner import query
from isaaclab.renderers.base_renderer import VisualMaterialWrite
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage

if TYPE_CHECKING:
    from .visual_material_cfg import VisualMaterialCfg

# note: this module may import ``pxr`` at the top because nothing loads it before Kit launches:
# the cfg references :class:`VisualMaterial` through a lazily-resolved string, and the event-term
# module defers its imports to construction/call time. It is instantiated only at scene creation.


@dataclass(frozen=True)
class _ChannelSpec:
    """Resolved binding of a logical channel to one shader input.

    ``semantic`` tells detached renderers how to mirror the write: ``"color"`` reaches Newton's
    ``model.shape_color``, ``"texture"`` values are asset paths swapped against a registered pool,
    and ``"scalar"``/``"float2"``/``"float3"`` reach attribute-writing backends only (OVRTX).
    """

    input_name: str
    """Base name of the shader input (no ``inputs:`` prefix)."""
    sdf_type_name: str
    """Name of the ``Sdf.ValueTypeNames`` entry for the input."""
    semantic: str
    """One of ``"color"``, ``"float3"``, ``"scalar"``, ``"float2"``, ``"texture"``."""
    default: Any
    """Value authored at init when the input has no authored value yet."""


def _omni_pbr_channels(shader: UsdShade.Shader) -> dict[str, _ChannelSpec]:
    """Channel table for the OmniPBR MDL family."""
    textured = bool(shader.GetInput("diffuse_texture") and shader.GetInput("diffuse_texture").Get())
    # on a textured OmniPBR the albedo comes from the texture and ``diffuse_color_constant`` is
    # ignored by the shader; ``diffuse_tint`` multiplies over the sampled texture instead.
    color_input = "diffuse_tint" if textured else "diffuse_color_constant"
    color_default = (1.0, 1.0, 1.0) if textured else (0.18, 0.18, 0.18)
    return {
        "color": _ChannelSpec(color_input, "Color3f", "color", color_default),
        "roughness": _ChannelSpec("reflection_roughness_constant", "Float", "scalar", 0.5),
        "metallic": _ChannelSpec("metallic_constant", "Float", "scalar", 0.0),
        "specular": _ChannelSpec("specular_level", "Float", "scalar", 0.5),
        "emissive_color": _ChannelSpec("emissive_color", "Color3f", "float3", (0.0, 0.0, 0.0)),
        "emissive_intensity": _ChannelSpec("emissive_intensity", "Float", "scalar", 0.0),
        "opacity": _ChannelSpec("opacity_constant", "Float", "scalar", 1.0),
        "uv_scale": _ChannelSpec("texture_scale", "Float2", "float2", (1.0, 1.0)),
        "uv_offset": _ChannelSpec("texture_translate", "Float2", "float2", (0.0, 0.0)),
        "uv_rotate": _ChannelSpec("texture_rotate", "Float", "scalar", 0.0),
        "texture": _ChannelSpec("diffuse_texture", "Asset", "texture", None),
    }


_OMNI_GLASS_CHANNELS = {
    "color": _ChannelSpec("glass_color", "Color3f", "color", (1.0, 1.0, 1.0)),
    "roughness": _ChannelSpec("frosting_roughness", "Float", "scalar", 0.0),
    "ior": _ChannelSpec("glass_ior", "Float", "scalar", 1.491),
}
"""Channel table for the OmniGlass MDL family."""

_PREVIEW_SURFACE_CHANNELS = {
    "color": _ChannelSpec("diffuseColor", "Color3f", "color", (0.18, 0.18, 0.18)),
    "roughness": _ChannelSpec("roughness", "Float", "scalar", 0.5),
    "metallic": _ChannelSpec("metallic", "Float", "scalar", 0.0),
    "emissive_color": _ChannelSpec("emissiveColor", "Color3f", "float3", (0.0, 0.0, 0.0)),
    "opacity": _ChannelSpec("opacity", "Float", "scalar", 1.0),
}
"""Channel table for ``UsdPreviewSurface`` shaders."""


class BaseVisualMaterial:
    """A visual material declared as a scene entity.

    The entity operates in one of two granularities, selected by the configured prim path:

    * **Bucket** (default): the entity wraps exactly one ``UsdShade.Material`` prim at a
      configuration-known absolute path outside the cloned environment namespace (e.g.
      ``/World/Materials/bucket_a``) so cloning never duplicates the prim: geometry in every
      environment that binds it (see
      :attr:`~isaaclab.sim.spawners.spawner_cfg.RigidObjectSpawnerCfg.visual_material_path` with an
      absolute path) shares this single material on all rendering backends. Writing a channel
      therefore restyles all bound geometry in all environments at once — the "bucket" granularity
      used by :func:`isaaclab.envs.mdp.randomize_visual_material`.
    * **Per-environment**: a prim path starting with ``{ENV_REGEX_NS}`` (e.g.
      ``{ENV_REGEX_NS}/Materials/style``) declares one *logical* material spawned in the source
      environment and replicated into every environment by the scene cloner. Geometry binds its own
      environment's clone (bind with the same ``{ENV_REGEX_NS}`` token in
      ``visual_material_path``/``visual_material_bindings``), so writes address environments
      independently: :meth:`write_channels` takes one value per environment and honors ``env_ids``
      — :func:`isaaclab.envs.mdp.randomize_visual_color` honors ``env_ids`` for them. The material
      is clone-planned and replicated like any other env-scoped spawn, and the clone prim paths
      are derived from the published :class:`~isaaclab.cloner.ClonePlan` (never from stage
      traversal). The whole-environment clone re-anchors the binding to each environment through
      ``Sdf.CopySpec`` (which remaps relationship targets inside the copied subtree); the Newton
      renderer resolves per-environment bindings through its own per-shape material map instead of
      the USD stage.

    On construction, the material is spawned from :attr:`VisualMaterialCfg.spawn` (or looked up when
    ``spawn`` is None) and its connected surface shader is resolved by following the material's
    surface output — a single connection hop, never a stage traversal. Each requested channel in
    :attr:`VisualMaterialCfg.channels` is then resolved to one shader input and authored with its
    current or default value (for per-environment materials this happens on the source prim before
    replication, so every clone starts with the authored defaults). Authoring at init is a hard
    requirement for detached renderers (OVRTX): their attribute writes cannot create attributes
    that were absent from the exported stage. Supported shader families:

    * ``UsdPreviewSurface``
    * OmniPBR-family MDL (``sourceAsset`` sub-identifier starting with ``OmniPBR``); the ``color``
      channel resolves to ``diffuse_tint`` when a ``diffuse_texture`` is authored (multiplies over
      the texture) and to ``diffuse_color_constant`` otherwise.
    * OmniGlass-family MDL (``color`` → ``glass_color``, ``roughness`` → ``frosting_roughness``,
      ``ior`` → ``glass_ior``).
    """

    cfg: VisualMaterialCfg
    """The configuration instance."""

    def __init__(self, cfg: VisualMaterialCfg):
        """Spawn (if configured) and resolve the material's writable channels.

        Args:
            cfg: The configuration instance.

        Raises:
            ValueError: If the prim path is not a concrete absolute path (after token
                substitution), a concrete path lies inside the cloned environment namespace
                without being clone-planned, the material's shader is missing or unsupported, or
                a requested channel is not available on the shader family.
        """

        self.cfg = cfg.copy()

        # per-environment material: clone planning resolved its prototype path (scene flow), or it
        # is declared with the token, which resolves to the source environment (standalone flow)
        planned = getattr(self.cfg.spawn, "spawn_path", None) if self.cfg.spawn is not None else None
        has_token = self.cfg.prim_path.startswith("{ENV_REGEX_NS}/")
        self._is_per_env = planned is not None or has_token
        if self._is_per_env:
            if planned:
                self.cfg.prim_path = planned
            else:
                self.cfg.prim_path = self.cfg.prim_path.format(ENV_REGEX_NS="/World/envs/env_0")
        elif self.cfg.prim_path.startswith("/World/envs/"):
            raise ValueError(
                f"Visual material prim path '{self.cfg.prim_path}' lies inside the cloned environment"
                " namespace but was not clone-planned (no spawn). Declare a per-environment material"
                " with an '{ENV_REGEX_NS}/...' path and a spawn, or move the material outside the"
                " environment namespace."
            )
        # clone paths and per-clone shader inputs are derived lazily from the published clone plan:
        # the entity is constructed before replication, so the clones do not exist yet
        self._env_material_paths: list[str] | None = None
        self._env_shader_paths_cached: list[str] | None = None
        self._env_inputs: dict[str, list[UsdShade.Input]] = {}

        prim_path = self.cfg.prim_path
        if (
            not Sdf.Path.IsValidPathString(prim_path)
            or not Sdf.Path(prim_path).IsPrimPath()
            or not Sdf.Path(prim_path).IsAbsolutePath()
        ):
            raise ValueError(f"Visual material prim path '{prim_path}' must be one concrete absolute prim path.")

        if self.cfg.spawn is not None:
            self.cfg.spawn.func(prim_path, self.cfg.spawn)

        stage = get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Visual material prim '{prim_path}' does not exist and no spawn cfg was given.")
        material = UsdShade.Material(prim)
        if not material:
            raise ValueError(f"Visual material prim '{prim_path}' is not a UsdShade.Material.")

        shader = _connected_surface_shader(material)
        self._shader = shader

        family_channels = _shader_family_channels(shader)
        requested = tuple(self.cfg.channels)
        if "texture" in requested:
            if not self.cfg.texture_pool:
                raise ValueError(
                    f"Visual material '{prim_path}' requests the 'texture' channel but declares no"
                    " 'texture_pool'. The pool is what guarantees every candidate texture is loaded"
                    " up front, so texture swaps stay free of runtime I/O."
                )
            # a texture-channel material starts textured (first pool entry) if the shader has no
            # authored texture yet; author it before resolving the other channels so the color
            # channel resolves to the over-texture input (e.g. OmniPBR diffuse_tint)
            texture_spec = family_channels.get("texture")
            if texture_spec is not None:
                texture_input = shader.CreateInput(
                    texture_spec.input_name, getattr(Sdf.ValueTypeNames, texture_spec.sdf_type_name)
                )
                if not texture_input.Get():
                    texture_input.Set(self.cfg.texture_pool[0])
                family_channels = _shader_family_channels(shader)

        self._channels: dict[str, _ChannelSpec] = {}
        self._inputs: dict[str, UsdShade.Input] = {}
        for name in requested:
            spec = family_channels.get(name)
            if spec is None:
                raise ValueError(
                    f"Visual material '{prim_path}' shader does not support channel '{name}'."
                    f" Available channels: {sorted(family_channels)}."
                )
            self._channels[name] = spec
            self._inputs[name] = _author_channel_input(shader, spec)

        # texture-pool declaration reaches detached renderers so they can preload the pool
        if self.cfg.texture_pool:
            sim = SimulationContext.instance()
            if sim is not None:
                sim.render_context.register_visual_material_textures(list(self.cfg.texture_pool))

    """
    Properties.
    """

    @property
    def material_prim_path(self) -> str:
        """Absolute path of the wrapped ``UsdShade.Material`` prim."""
        return self.cfg.prim_path

    @property
    def _shader_prim_path(self) -> str:
        """Absolute path of the connected surface shader prim."""
        return str(self._shader.GetPrim().GetPath())

    @property
    def channels(self) -> tuple[str, ...]:
        """Names of the resolved writable channels."""
        return tuple(self._channels)

    @property
    def is_per_env(self) -> bool:
        """Whether this is a per-environment material (one clone per environment)."""
        return self._is_per_env

    @property
    def env_material_paths(self) -> list[str]:
        """Absolute paths of the per-environment material clones, indexed by environment id.

        Derived from the published :class:`~isaaclab.cloner.ClonePlan` on first access, so it is
        only valid after scene replication has run (event terms always fire later than that).
        """
        if self._env_material_paths is None:
            self._env_material_paths = self._expand_source_path(self.material_prim_path)
        return self._env_material_paths

    @property
    def _env_shader_paths(self) -> list[str]:
        """Absolute paths of the per-environment shader clones, indexed by environment id."""
        if self._env_shader_paths_cached is None:
            self._env_shader_paths_cached = self._expand_source_path(self._shader_prim_path)
        return self._env_shader_paths_cached

    """
    Operations.
    """

    @classmethod
    def write_channels(
        cls,
        materials: Sequence[BaseVisualMaterial],
        channels: dict[str, torch.Tensor | list[str]],
        env_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> None:
        """Write channel values on the materials and notify all camera renderers once.

        The USD stage is the authoritative representation: values are authored on each material's
        shader inputs, which Kit-based renderers consume through regular change processing.
        Detached renderers (OVRTX, Newton) are synchronized through
        :meth:`~isaaclab.renderers.render_context.RenderContext.notify_visual_material_written`.

        A classmethod so post-launch callers holding entity instances (event terms) need no
        import of this module; call it as ``materials[0].write_channels(materials, ...)``.

        Args:
            materials: The material entities to write. All bucket or all per-environment.
            channels: Mapping of channel name to values. For bucket materials, a float tensor of
                shape (len(materials), c) for numeric channels, or a ``list[str]`` of asset paths
                of length len(materials) for the ``texture`` channel. For per-environment
                materials, a float tensor of shape (len(materials), len(env_ids), c) for numeric
                channels, or a list of per-material lists of asset paths, shaped
                (len(materials), len(env_ids)), for the ``texture`` channel.
            env_ids: Environments to write, for per-environment materials only. None writes every
                environment. Bucket writes are global by construction and reject ``env_ids``.

        Raises:
            ValueError: If a channel's values have the wrong shape, a material does not resolve
                the channel, bucket and per-environment materials are mixed, or ``env_ids`` is
                passed for bucket materials.
        """
        if len(materials) == 0 or len(channels) == 0:
            return

        per_env = materials[0].is_per_env
        if any(material.is_per_env != per_env for material in materials):
            raise ValueError("Cannot mix bucket and per-environment materials in one write_channels call.")
        if per_env:
            cls._write_per_env_channels(materials, channels, env_ids)
        elif env_ids is not None:
            raise ValueError(
                "env_ids is only supported for per-environment materials; bucket material writes are"
                " global to every environment binding the bucket."
            )
        else:
            cls._write_bucket_channels(materials, channels)

    @classmethod
    def _write_bucket_channels(
        cls, materials: Sequence[BaseVisualMaterial], channels: dict[str, torch.Tensor | list[str]]
    ) -> None:
        """Author bucket channel values on each material's shader inputs and notify renderers."""
        writes: list[VisualMaterialWrite] = []
        for name, values in channels.items():
            specs = _resolve_channel_specs(materials, name)
            if specs[0].semantic == "texture":
                if len(values) != len(materials):
                    raise ValueError(f"Channel '{name}' expects {len(materials)} asset paths; got {len(values)}.")
                for material, path in zip(materials, values, strict=True):
                    material._inputs[name].Set(path)
            else:
                values = _validated_bucket_values(values, len(materials), name)
                for material, spec, row in zip(materials, specs, values, strict=True):
                    _set_shader_input(material._inputs[name], spec.sdf_type_name, row)
            writes += _bucket_writes(materials, specs, values)
        _notify_renderers(writes)

    @classmethod
    def _write_per_env_channels(
        cls,
        materials: Sequence[BaseVisualMaterial],
        channels: dict[str, torch.Tensor | list[str]],
        env_ids: torch.Tensor | Sequence[int] | None,
    ) -> None:
        """Author per-environment channel values on the selected clones and notify renderers.

        Values are authored only on the selected environments' clone shaders; the emitted
        :class:`~isaaclab.renderers.base_renderer.VisualMaterialWrite` rows carry the matching
        per-clone material/shader paths so detached renderers restyle the same subset.
        """
        num_envs = len(materials[0].env_material_paths)
        selected = list(range(num_envs)) if env_ids is None else [int(env_id) for env_id in env_ids]

        writes: list[VisualMaterialWrite] = []
        for name, values in channels.items():
            specs = _resolve_channel_specs(materials, name)
            if specs[0].semantic == "texture":
                _validate_per_env_texture_paths(values, len(materials), len(selected), name)
                for material, material_paths in zip(materials, values, strict=True):
                    inputs = material._env_channel_inputs(name)
                    for env_id, path in zip(selected, material_paths, strict=True):
                        inputs[env_id].Set(path)
            else:
                values = _validated_per_env_values(values, len(materials), len(selected), name)
                for material, spec, material_rows in zip(materials, specs, values, strict=True):
                    inputs = material._env_channel_inputs(name)
                    for env_id, row in zip(selected, material_rows, strict=True):
                        _set_shader_input(inputs[env_id], spec.sdf_type_name, row)
            writes += _per_env_writes(materials, specs, values, selected)
        _notify_renderers(writes)

    def _expand_source_path(self, path: str) -> list[str]:
        """Derive one cloned path per environment for a source-space path, from the clone plan.

        Raises:
            RuntimeError: If no clone plan has been published yet.
            ValueError: If the material is not per-environment, or no plan row replicates the
                path to every environment (heterogeneous clone layouts).
        """
        if not self._is_per_env:
            raise ValueError(f"Visual material '{self.material_prim_path}' is a bucket material; it has no clones.")
        sim = SimulationContext.instance()
        plan = sim.get_clone_plan() if sim is not None else None
        if plan is None:
            raise RuntimeError(
                f"Per-environment visual material '{self.material_prim_path}' derives its clone paths from"
                " the published clone plan, which is not available yet. Write only after scene replication."
            )
        # Callers index the result positionally (``env_material_paths[env_id]``), so the plan must
        # replicate the path to every environment under the contiguous ``0..num_envs - 1`` ids.
        env_ids = query.path_env_ids(plan, path)
        if env_ids == tuple(range(plan.clone_mask.shape[1])):
            clones = [clone for env_id in env_ids if (clone := query.path_to_clone(plan, path, env_id)) is not None]
            if len(clones) == len(env_ids):
                return clones
        raise ValueError(
            f"Per-environment visual material '{self.material_prim_path}' is not replicated to every"
            " environment by one clone-plan row. The material must be declared with a spawn inside the"
            " cloned environment namespace so clone planning assigns it a row (materials wrapping"
            " pre-existing prims with spawn=None are not replicated)."
        )

    def _env_channel_inputs(self, channel: str) -> list[UsdShade.Input]:
        """Return one shader input handle per environment clone for the channel (lazily resolved)."""
        inputs = self._env_inputs.get(channel)
        if inputs is None:
            spec = self._channels[channel]
            stage = get_current_stage()
            inputs = []
            for shader_path in self._env_shader_paths:
                shader = UsdShade.Shader(stage.GetPrimAtPath(shader_path))
                if not shader:
                    raise ValueError(
                        f"Per-environment shader clone '{shader_path}' does not exist on the stage."
                        " Was the source environment replicated?"
                    )
                inputs.append(shader.CreateInput(spec.input_name, getattr(Sdf.ValueTypeNames, spec.sdf_type_name)))
            self._env_inputs[channel] = inputs
        return inputs

    def read_channel(self, channel: str, env_id: int | None = None) -> Any:
        """Return the authored value of a channel's shader input, or None when not authored.

        Args:
            channel: The channel name.
            env_id: For per-environment materials, the environment whose clone to read. None reads
                the source-environment prim (the only prim of a bucket material).

        Raises:
            ValueError: If ``env_id`` is passed for a bucket material.
        """
        if env_id is not None:
            if not self._is_per_env:
                raise ValueError(
                    f"Visual material '{self.material_prim_path}' is a bucket material shared by all"
                    " environments; read_channel does not take an env_id."
                )
            return self._env_channel_inputs(channel)[env_id].Get()
        return self._inputs[channel].Get()


def _shader_family_channels(shader: UsdShade.Shader) -> dict[str, _ChannelSpec]:
    """Return the channel table for the shader's family, validating the family is supported."""
    shader_path = shader.GetPrim().GetPath()
    implementation = shader.GetImplementationSource()
    if implementation == UsdShade.Tokens.id:
        shader_id = shader.GetShaderId()
        if shader_id == "UsdPreviewSurface":
            return _PREVIEW_SURFACE_CHANNELS
        raise ValueError(f"Visual material shader '{shader_path}' has unsupported id '{shader_id}'.")
    if implementation == UsdShade.Tokens.sourceAsset:
        sub_identifier = shader.GetSourceAssetSubIdentifier("mdl")
        if sub_identifier and sub_identifier.startswith("OmniPBR"):
            return _omni_pbr_channels(shader)
        if sub_identifier and sub_identifier.startswith("OmniGlass"):
            return _OMNI_GLASS_CHANNELS
        raise ValueError(
            f"Visual material shader '{shader_path}' has unsupported MDL sub-identifier '{sub_identifier}'."
            " Only the OmniPBR family exposes randomization channels on every rendering backend."
        )
    raise ValueError(f"Visual material shader '{shader_path}' has unsupported implementation source.")


def _set_shader_input(shader_input: UsdShade.Input, sdf_type_name: str, value) -> None:
    """Set one shader input from a float sequence (or scalar), converting to its Gf value type."""

    if sdf_type_name == "Color3f":
        shader_input.Set(Gf.Vec3f(float(value[0]), float(value[1]), float(value[2])))
    elif sdf_type_name == "Float2":
        shader_input.Set(Gf.Vec2f(float(value[0]), float(value[1])))
    else:
        shader_input.Set(float(value if not hasattr(value, "__len__") else value[0]))


def _resolve_channel_specs(materials: Sequence[BaseVisualMaterial], name: str) -> list[_ChannelSpec]:
    """Resolve one channel to its spec on every material, failing on unresolved channels."""
    specs = []
    for material in materials:
        spec = material._channels.get(name)
        if spec is None:
            raise ValueError(f"Material '{material.material_prim_path}' does not resolve channel '{name}'.")
        specs.append(spec)
    return specs


def _validated_bucket_values(values: torch.Tensor, num_materials: int, name: str) -> torch.Tensor:
    """Normalize numeric bucket values to a CPU float32 tensor of shape (num_materials, c)."""
    values = values.detach().to(device="cpu", dtype=torch.float32)
    if values.dim() == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] != num_materials:
        raise ValueError(f"Channel '{name}' expects {num_materials} rows; got {values.shape[0]}.")
    return values


def _validated_per_env_values(values: torch.Tensor, num_materials: int, num_selected: int, name: str) -> torch.Tensor:
    """Normalize numeric per-env values to a CPU float32 tensor of shape (num_materials, num_selected, c)."""
    values = values.detach().to(device="cpu", dtype=torch.float32)
    if values.dim() == 2:
        values = values.unsqueeze(-1)
    if values.dim() != 3 or values.shape[0] != num_materials or values.shape[1] != num_selected:
        raise ValueError(
            f"Channel '{name}' expects values shaped (len(materials), len(env_ids), c) ="
            f" ({num_materials}, {num_selected}, c); got {tuple(values.shape)}."
        )
    return values


def _validate_per_env_texture_paths(values: list[list[str]], num_materials: int, num_selected: int, name: str) -> None:
    """Validate per-env texture values: one asset path per material per selected environment."""
    if len(values) != num_materials or any(len(paths) != num_selected for paths in values):
        raise ValueError(
            f"Channel '{name}' expects one asset path per material per environment, shaped"
            f" (len(materials), len(env_ids)) = ({num_materials}, {num_selected})."
        )


def _rows_by_resolved_attr(specs: list[_ChannelSpec]) -> dict[str, list[int]]:
    """Group material indices by resolved attribute name.

    One write per resolved attribute: the same channel can resolve differently per material
    (e.g. color -> diffuse_tint on textured OmniPBR, diffuse_color_constant else).
    """
    rows_by_attr: dict[str, list[int]] = {}
    for i, spec in enumerate(specs):
        rows_by_attr.setdefault(f"inputs:{spec.input_name}", []).append(i)
    return rows_by_attr


def _bucket_writes(
    materials: Sequence[BaseVisualMaterial], specs: list[_ChannelSpec], values
) -> list[VisualMaterialWrite]:
    """Build the notify rows for one bucket channel write: one row per material."""
    writes = []
    for attr_name, rows in _rows_by_resolved_attr(specs).items():
        writes.append(
            VisualMaterialWrite(
                material_paths=[materials[i].material_prim_path for i in rows],
                shader_paths=[materials[i]._shader_prim_path for i in rows],
                attr_name=attr_name,
                semantic=specs[rows[0]].semantic,
                values=[values[i] for i in rows] if isinstance(values, list) else values[rows],
            )
        )
    return writes


def _per_env_writes(
    materials: Sequence[BaseVisualMaterial], specs: list[_ChannelSpec], values, selected: list[int]
) -> list[VisualMaterialWrite]:
    """Build the notify rows for one per-env channel write: the (material, selected env) product."""
    writes = []
    for attr_name, rows in _rows_by_resolved_attr(specs).items():
        if specs[rows[0]].semantic == "texture":
            row_values = [values[i][j] for i in rows for j in range(len(selected))]
        else:
            row_values = values[rows].reshape(len(rows) * len(selected), -1)
        writes.append(
            VisualMaterialWrite(
                material_paths=[materials[i].env_material_paths[env_id] for i in rows for env_id in selected],
                shader_paths=[materials[i]._env_shader_paths[env_id] for i in rows for env_id in selected],
                attr_name=attr_name,
                semantic=specs[rows[0]].semantic,
                values=row_values,
            )
        )
    return writes


def _notify_renderers(writes: list[VisualMaterialWrite]) -> None:
    """Broadcast the writes to every camera renderer through the render context."""
    sim = SimulationContext.instance()
    if sim is not None:
        sim.render_context.notify_visual_material_written(writes)


def _author_channel_input(shader: UsdShade.Shader, spec: _ChannelSpec) -> UsdShade.Input:
    """Create (if needed) and default-author one channel's shader input, returning the handle.

    Detached renderers export the stage once; attributes absent at export time cannot be written
    afterwards, so every writable channel must hold an authored value before the first render.
    """

    shader_input = shader.CreateInput(spec.input_name, getattr(Sdf.ValueTypeNames, spec.sdf_type_name))
    if shader_input.Get() is None and spec.default is not None:
        _set_shader_input(shader_input, spec.sdf_type_name, spec.default)
    return shader_input


def _connected_surface_shader(material: UsdShade.Material) -> UsdShade.Shader:
    """Follow the material's surface output one hop to its connected shader."""

    outputs = (
        material.GetSurfaceOutput("mdl"),
        material.GetSurfaceOutput(),
        material.GetOutput("mdl:surface"),
        material.GetOutput("surface"),
    )
    connected = next(
        (output.GetConnectedSource() for output in outputs if output and output.HasConnectedSource()), None
    )
    material_path = material.GetPrim().GetPath()
    if connected is None:
        raise ValueError(f"Visual material '{material_path}' has no connected surface shader.")
    shader = UsdShade.Shader(connected[0].GetPrim())
    if not shader:
        raise ValueError(
            f"Visual material '{material_path}' surface source '{connected[0].GetPrim().GetPath()}' is not a shader."
        )
    return shader
