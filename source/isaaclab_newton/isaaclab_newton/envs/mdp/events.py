# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton implementations of visual event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from ...assets.visual_material.shape_writer import scatter_srgb_shape_colors
from ...physics.newton_manager import NewtonManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


__all__ = ["randomize_visual_shape"]


class randomize_visual_shape(ManagerTermBase):
    """Newton implementation of per-environment visual randomization.

    Writes the Newton model's per-shape visual state directly: the model stores one value
    per shape (``model.shape_color``), so per-environment writes are index-addressed tensor
    writes — one sampled value per environment, scattered to the asset's shape rows of that
    environment with one kernel per channel. Shape rows are resolved once from
    ``model.shape_label`` by environment prim prefix.

    Channel support follows the model's storage, not the term's interface: ``color`` is
    writable today; channels without per-shape storage raise at init with the reason.
    """

    _SUPPORTED_CHANNELS = ("color",)
    """Channels with per-shape Newton-model storage today.

    ``texture`` joins once the renderer exposes a public per-shape texture-id write path to
    event terms; scalar response channels join with the upstream per-material table.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset = env.scene[asset_cfg.name]
        self._body_names = asset_cfg.body_names

        # fail fast on channels the model cannot represent per shape
        for name in cfg.params["channels"]:
            if name not in self._SUPPORTED_CHANNELS:
                raise NotImplementedError(
                    f"Per-environment visual randomization of channel '{name}' is not supported:"
                    f" the Newton model stores {list(self._SUPPORTED_CHANNELS)} per shape."
                    " 'texture' requires a public per-shape texture-id write path on the renderer;"
                    " scalar response channels require the upstream per-material table"
                    " (shape_material_id + material tables). Use the bucket-granular"
                    " 'randomize_visual_material' term for those channels."
                )

        # compile each channel spec once (same grammar as randomize_visual_material)
        self._samplers: list[tuple[str, str, Any]] = []
        for name, spec in cfg.params["channels"].items():
            if isinstance(spec, dict) and "choices" in spec:
                values = torch.tensor(list(spec["choices"]), dtype=torch.float32).reshape(len(spec["choices"]), -1)
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

        # per-environment shape rows are resolved from the model on first fire (the Newton
        # model does not exist yet when event terms construct)
        self._rows_per_env: list[torch.Tensor] | None = None

    def _resolve_rows(self, env: ManagerBasedEnv) -> list[torch.Tensor]:
        """Group the asset's ``model.shape_label`` rows by environment (memoized)."""
        if self._rows_per_env is not None:
            return self._rows_per_env

        model = NewtonManager.get_model()
        if model is None:
            raise RuntimeError("Per-environment visual randomization requires an initialized Newton model.")

        # "/World/envs/env_.*/Robot" -> per-env concrete prefixes via the scene's clone namespace
        template = self._asset.cfg.prim_path
        regex_ns = env.scene.cloner_cfg.clone_regex
        if not template.startswith(regex_ns):
            raise ValueError(
                f"Asset prim path '{template}' is not per-environment (expected it under '{regex_ns}');"
                " per-environment visual randomization requires a cloned asset."
            )
        asset_subpath = template[len(regex_ns) :]
        prefixes = [
            (
                f"{env_path}{asset_subpath}",
                tuple(f"{env_path}{asset_subpath}/{body}" for body in self._body_names or ()),
            )
            for env_path in env.scene.env_prim_paths
        ]

        device = wp.device_to_torch(model.device)
        rows_per_env: list[list[int]] = [[] for _ in prefixes]
        for row, label in enumerate(model.shape_label):
            for env_id, (asset_prefix, body_prefixes) in enumerate(prefixes):
                if not label.startswith(asset_prefix + "/"):
                    continue
                if not body_prefixes or any(
                    label == body_prefix or label.startswith(body_prefix + "/") for body_prefix in body_prefixes
                ):
                    rows_per_env[env_id].append(row)
                break

        self._rows_per_env = [torch.tensor(rows, dtype=torch.int32, device=device) for rows in rows_per_env]
        return self._rows_per_env

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg,
        channels: dict[str, tuple | dict],
    ):
        # specs are compiled at init; rows are per environment, so env_ids is honored
        del asset_cfg, channels

        rows_per_env = self._resolve_rows(env)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = range(len(rows_per_env))[env_ids] if isinstance(env_ids, slice) else range(len(rows_per_env))
        else:
            env_ids = env_ids.tolist()
        if len(env_ids) == 0:
            return

        model = NewtonManager.get_model()
        device = wp.device_to_torch(model.device)
        rows = torch.cat([rows_per_env[i] for i in env_ids])
        env_slot = torch.repeat_interleave(
            torch.arange(len(env_ids), device=device, dtype=torch.int32),
            torch.tensor([rows_per_env[i].numel() for i in env_ids], device=device),
        )

        for name, kind, data in self._samplers:
            if kind == "values":
                sampled = data[torch.randint(data.shape[0], (len(env_ids),))]
            else:
                low, high = data
                sampled = math_utils.sample_uniform(low, high, (len(env_ids), low.numel()), device="cpu")
            # one value per environment, scattered to every selected row of that environment
            wp.launch(
                scatter_srgb_shape_colors,
                dim=rows.numel(),
                inputs=[
                    wp.from_torch(sampled.to(device=device).contiguous(), dtype=wp.vec3),
                    wp.from_torch(env_slot.contiguous(), dtype=wp.int32),
                    wp.from_torch(rows.contiguous(), dtype=wp.int32),
                    model.shape_color,
                ],
                device=model.device,
            )
