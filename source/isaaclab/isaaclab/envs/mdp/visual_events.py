# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-neutral visual event terms."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import VisualMaterialCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.backend_utils import FactoryBase

if TYPE_CHECKING:
    from isaaclab.assets import VisualMaterial
    from isaaclab.envs import ManagerBasedEnv


__all__ = ["randomize_visual_color", "randomize_visual_material", "randomize_visual_shape"]


def _resolve_materials_granularity(materials: list[VisualMaterial]) -> bool:
    """Return whether the materials are per-environment, rejecting mixed declarations."""
    per_env = {material.is_per_env for material in materials}
    if len(per_env) > 1:
        raise ValueError(
            "Cannot mix bucket and per-environment materials in one event term; declare one term per granularity."
        )
    return per_env.pop()


def _resolve_env_selection(
    env: ManagerBasedEnv, env_ids: torch.Tensor | slice | None
) -> tuple[torch.Tensor | None, int]:
    """Normalize the event manager's env selection to (env_ids or None for all, selection count)."""
    if env_ids is None or isinstance(env_ids, slice):
        return None, env.scene.num_envs
    return env_ids, len(env_ids)


class randomize_visual_material(ManagerTermBase):
    """Randomize channels of visual materials.

    The generalization of :class:`randomize_visual_color` to every writable material channel.
    Each configured material is a scene-level :class:`~isaaclab.assets.VisualMaterial`, and the
    term follows the granularity the entity declares (see :class:`randomize_visual_color`):
    bucket materials get one sampled value per material per channel, written globally with
    ``env_ids`` ignored; per-environment materials (``{ENV_REGEX_NS}`` prim paths) get one value
    per material per environment, with ``env_ids`` honored. Every fire issues a single batched
    write (one USD author pass and one renderer notify).

    Channel sampling specifications, keyed by channel name:

    * ``(low, high)`` — uniform range. Scalars use floats; vector channels (``color``,
      ``uv_scale``, ...) use tuples of matching length for ``low``/``high``.
    * ``{"r": (lo, hi), "g": (lo, hi), "b": (lo, hi)}`` — per-component color ranges.
    * ``{"choices": [...]}`` — uniform pick from a discrete set. This is how textures are
      randomized: entries are asset paths from the materials' declared
      :attr:`~isaaclab.assets.VisualMaterialCfg.texture_pool`. For per-environment materials,
      one path is drawn per material per environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        material_cfgs = cfg.params["materials"]
        if isinstance(material_cfgs, SceneEntityCfg):
            material_cfgs = [material_cfgs]
        if not material_cfgs:
            raise ValueError("Visual material randomization requires at least one material in 'materials'.")

        self._materials: list[VisualMaterial] = []
        for material_cfg in material_cfgs:
            entity = env.scene[material_cfg.name]
            if not isinstance(getattr(entity, "cfg", None), VisualMaterialCfg):
                raise TypeError(
                    f"Visual material '{material_cfg.name}' must be a VisualMaterial; got {type(entity).__name__}."
                )
            self._materials.append(entity)
        self._per_env = _resolve_materials_granularity(self._materials)

        for channel in cfg.params["channels"]:
            for material in self._materials:
                if channel not in material.channels:
                    raise ValueError(
                        f"Material '{material.material_prim_path}' does not resolve channel '{channel}'."
                        f" Declare it in the entity's 'channels' (resolved: {list(material.channels)})."
                    )

        # compile each channel spec once into a sampler: per-fire work is sampling + one write
        self._samplers: list[tuple[str, str, Any]] = []
        for name, spec in cfg.params["channels"].items():
            if isinstance(spec, dict) and "choices" in spec:
                choices = list(spec["choices"])
                if isinstance(choices[0], str):  # asset paths, i.e. the texture channel
                    self._samplers.append((name, "paths", choices))
                else:
                    values = torch.tensor(choices, dtype=torch.float32).reshape(len(choices), -1)
                    self._samplers.append((name, "values", values))
            else:
                if isinstance(spec, dict):
                    low = [spec[key][0] for key in ("r", "g", "b")]
                    high = [spec[key][1] for key in ("r", "g", "b")]
                else:
                    low, high = spec
                low = torch.as_tensor(low, dtype=torch.float32).reshape(-1)
                high = torch.as_tensor(high, dtype=torch.float32).reshape(-1)
                self._samplers.append((name, "range", (low, high)))

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        materials: list[SceneEntityCfg] | SceneEntityCfg,
        channels: dict[str, tuple | dict],
    ):
        # specs are compiled at init
        del materials, channels
        num = len(self._materials)

        if not self._per_env:
            # bucket materials are shared, so writes are global by design
            del env, env_ids
            sampled: dict[str, torch.Tensor | list[str]] = {}
            for name, kind, data in self._samplers:
                if kind == "paths":
                    sampled[name] = [data[i] for i in torch.randint(len(data), (num,)).tolist()]
                elif kind == "values":
                    sampled[name] = data[torch.randint(data.shape[0], (num,))]
                else:
                    low, high = data
                    sampled[name] = math_utils.sample_uniform(low, high, (num, low.numel()), device="cpu")
            self._materials[0].write_channels(self._materials, sampled)
            return

        env_ids, num_selected = _resolve_env_selection(env, env_ids)
        sampled = {}
        for name, kind, data in self._samplers:
            if kind == "paths":
                sampled[name] = [
                    [data[j] for j in torch.randint(len(data), (num_selected,)).tolist()] for _ in range(num)
                ]
            elif kind == "values":
                sampled[name] = data[torch.randint(data.shape[0], (num, num_selected))]
            else:
                low, high = data
                sampled[name] = math_utils.sample_uniform(low, high, (num, num_selected, low.numel()), device="cpu")
        self._materials[0].write_channels(self._materials, sampled, env_ids=env_ids)


class randomize_visual_color(randomize_visual_material):
    """Randomize the colors of visual materials.

    :class:`randomize_visual_material` restricted to the ``color`` channel, with the sampling
    range passed as ``colors`` — two RGB tuples defining the range, or a dictionary with ``r``,
    ``g``, and ``b`` keys whose values are ``(low, high)`` ranges. Granularity follows the
    material declaration exactly as in the parent term: bucket materials restyle globally
    (``env_ids`` ignored), per-environment materials sample per environment (``env_ids``
    honored).
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        cfg = copy.copy(cfg)
        cfg.params = {"materials": cfg.params["materials"], "channels": {"color": cfg.params["colors"]}}
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        materials: list[SceneEntityCfg] | SceneEntityCfg,
        colors: tuple[tuple[float, float, float], tuple[float, float, float]] | dict[str, tuple[float, float]],
    ):
        # the event manager passes the original params; sampling specs were compiled at init
        del materials, colors
        super().__call__(env, env_ids, materials=None, channels=None)


class randomize_visual_shape(FactoryBase, ManagerTermBase):
    """Randomize visual channels per shape, per environment (Newton-only capability).

    This term randomizes *within* a material: it writes per-shape visual state below the
    material binding, so shapes sharing one material diverge — which no material write can
    express (writing a material restyles everything bound to it, on every backend). The Newton
    model stores per-shape colors (``model.shape_color``), making this representable; backends
    whose unit of appearance is the material prim (PhysX/Kit, Omniverse PhysX) cannot represent
    it and raise at construction with the reason. For per-environment granularity that works on
    every backend, use per-environment :class:`~isaaclab.assets.VisualMaterial` entities with
    :class:`randomize_visual_color` / :class:`randomize_visual_material` instead.

    ``env_ids`` is honored: partial resets restyle only the environments that reset. The
    sampling grammar matches :class:`randomize_visual_material`; channels the backend cannot
    represent per shape raise at construction.
    """

    def __new__(cls, cfg: EventTermCfg, env: ManagerBasedEnv) -> ManagerTermBase:
        """Create the backend-specific implementation of the term."""
        return super().__new__(cls, cfg, env)

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Resolve targets and compile sampling specs (implemented per backend).

        Implementations resolve ``cfg.params["asset_cfg"]`` against the scene, validate that
        every channel in ``cfg.params["channels"]`` has per-shape storage on this backend
        (raising :class:`NotImplementedError` naming the missing capability otherwise), and
        compile the channel specs so a fire is sampling plus one batched write. Backends with
        no per-environment visual representation raise here unconditionally.
        """
        raise NotImplementedError("Constructed through the factory; implemented by the backend class.")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        channels: dict[str, tuple | dict],
    ):
        """Sample one value per environment per channel and write the backend's per-shape state.

        Args:
            env: The environment instance.
            env_ids: Environments to restyle; honored, so partial resets restyle only the
                environments that reset. None restyles all environments.
            asset_cfg: The scene asset whose shapes to restyle, optionally narrowed with
                ``body_names``. (Resolved at init; passed per fire by the event manager.)
            channels: Channel sampling specs, same grammar as
                :class:`randomize_visual_material`. (Compiled at init; passed per fire by the
                event manager.)
        """
        raise NotImplementedError("Constructed through the factory; implemented by the backend class.")


class _randomize_visual_shape_unsupported(ManagerTermBase):
    """Raising stub for backends whose unit of appearance is the material prim."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        raise NotImplementedError(
            "Per-shape visual randomization is not representable on this backend: shapes have no"
            " color storage of their own (the material prim is the unit of appearance). Use"
            " per-environment visual materials with 'randomize_visual_color', or the"
            " bucket-granular 'randomize_visual_material'."
        )


# Kit-family backends have no per-shape visual storage; register the raising stub inline instead
# of shipping passthrough modules in the backend packages.
randomize_visual_shape.register("physx", _randomize_visual_shape_unsupported)
randomize_visual_shape.register("ovphysx", _randomize_visual_shape_unsupported)
