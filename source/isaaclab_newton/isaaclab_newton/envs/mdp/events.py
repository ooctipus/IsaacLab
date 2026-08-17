# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-specific event terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from isaaclab_newton.assets import RigidObject


def randomize_rigid_body_mesh(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Select a uniformly random precompiled mesh variant at reset.

    The variant order matches the asset's multi-asset spawner. This changes Newton
    MJWarp collision geometry and inertia; USD/Fabric visuals remain unchanged.

    Args:
        env: The environment.
        env_ids: Environment indices. If ``None``, randomize every environment.
        asset_cfg: Rigid object to randomize.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if asset.num_mesh_variants == 0:
        raise ValueError(f"Rigid object {asset_cfg.name!r} does not enable mesh variants.")
    if env_ids is None:
        env_ids = torch.arange(asset.num_instances, device=asset.device)
    variant_ids = torch.randint(asset.num_mesh_variants, (env_ids.shape[0],), device=asset.device, dtype=torch.int32)
    asset.write_mesh_variant_to_sim(variant_ids, env_ids)
