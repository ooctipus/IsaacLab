# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.sim.spawners.materials import VisualMaterialCfg as VisualMaterialSpawnerCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .visual_material import VisualMaterial


@configclass
class VisualMaterialCfg:
    """Configuration for a scene-level visual material entity.

    The material is declared like any other scene entity in an
    :class:`~isaaclab.scene.InteractiveSceneCfg` and addressed through
    :class:`~isaaclab.managers.SceneEntityCfg`. The prim path selects the granularity:

    * A concrete absolute path outside the cloned environment namespace (e.g.
      ``/World/Materials/bucket_a``) declares a **bucket**: geometry in every environment shares
      the same material prim (bind it with an absolute
      :attr:`~isaaclab.sim.spawners.spawner_cfg.RigidObjectSpawnerCfg.visual_material_path`).
    * A path starting with ``{ENV_REGEX_NS}`` (e.g. ``{ENV_REGEX_NS}/Materials/style``) declares a
      **per-environment** material: clone-planned and replicated like any other env-scoped spawn,
      and addressed per environment (bind geometry with the same token so each environment binds
      its own clone).
    """

    class_type: type["VisualMaterial"] | str = "{DIR}.visual_material:VisualMaterial"
    """The associated entity class (resolved lazily so composing configs never imports it)."""

    prim_path: str = MISSING
    """Prim path of the material.

    Either a concrete absolute path outside ``/World/envs`` (bucket material, e.g.
    ``/World/Materials/bucket_a``) or a ``{ENV_REGEX_NS}``-prefixed path (per-environment
    material, e.g. ``{ENV_REGEX_NS}/Materials/style``). No other regex is allowed.
    """

    spawn: VisualMaterialSpawnerCfg | None = MISSING
    """Spawn configuration for the material (e.g. :class:`~isaaclab.sim.spawners.materials.PbrMdlCfg`,
    :class:`~isaaclab.sim.spawners.materials.PreviewSurfaceCfg`).

    Set to None to wrap a material prim that already exists on the stage.
    """

    channels: tuple[str, ...] = ("color",)
    """Writable channels to resolve and author on the material's shader. Defaults to color only.

    Channels map to shader inputs per family (see :class:`VisualMaterial`); e.g. for OmniPBR,
    ``color`` resolves to ``diffuse_tint`` on textured materials (modulates over the texture) and
    ``diffuse_color_constant`` otherwise, ``roughness`` to ``reflection_roughness_constant``, and
    ``uv_scale``/``uv_offset``/``uv_rotate`` to the texture transform inputs. Every requested
    channel is authored at construction so detached renderers can write it after their stage
    export.
    """

    texture_pool: tuple[str, ...] = ()
    """Candidate texture asset paths for the ``texture`` channel. Defaults to no pool.

    Declaring the pool up front is what keeps texture randomization free of runtime I/O: backends
    that swap by index (Newton) load every pool texture once at renderer initialization, and
    per-fire swaps then only re-point indices. Required when ``channels`` includes ``texture``.
    """
