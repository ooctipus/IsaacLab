# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event composition for the full-board Factory environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.events import randomize_rigid_body_material
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    import torch

    from isaaclab.envs import ManagerBasedEnv


class randomize_rigid_body_materials(ManagerTermBase):
    """Randomize material properties for a homogeneous group of scene assets."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset_cfgs: tuple[SceneEntityCfg, ...] = cfg.params["asset_cfgs"]
        material_params = {key: value for key, value in cfg.params.items() if key != "asset_cfgs"}
        self._terms = tuple(
            randomize_rigid_body_material(
                cfg.replace(func=randomize_rigid_body_material, params=material_params | {"asset_cfg": asset_cfg}),
                env,
            )
            for asset_cfg in self._asset_cfgs
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfgs: tuple[SceneEntityCfg, ...],
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
    ) -> None:
        for term, asset_cfg in zip(self._terms, self._asset_cfgs, strict=True):
            term(
                env,
                env_ids,
                static_friction_range,
                dynamic_friction_range,
                restitution_range,
                num_buckets,
                asset_cfg,
            )
