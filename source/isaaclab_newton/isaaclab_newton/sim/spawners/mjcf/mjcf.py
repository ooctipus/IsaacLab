# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab.sim.utils import clone, create_prim

if TYPE_CHECKING:
    from pxr import Usd

    from .mjcf_cfg import NewtonMjcfFileCfg

NEWTON_MJCF_ASSET_PATH_ATTR = "isaaclab:newton:mjcf:assetPath"
"""USD attribute containing the local MJCF asset path."""

NEWTON_MJCF_SELF_COLLISION_ATTR = "isaaclab:newton:mjcf:selfCollision"
"""USD attribute controlling native MJCF self-collision."""


@clone
def spawn_newton_mjcf(
    prim_path: str,
    cfg: NewtonMjcfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Author a lightweight marker for one Newton-native MJCF asset.

    Args:
        prim_path: USD path for the asset marker.
        cfg: Native MJCF loading configuration.
        translation: Marker translation relative to its parent [m].
        orientation: Marker orientation in ``(x, y, z, w)`` order.
        **kwargs: Unused spawner compatibility arguments.

    Returns:
        Marker prim consumed by the Newton model-builder import path.

    Raises:
        FileNotFoundError: If :attr:`NewtonMjcfFileCfg.asset_path` does not
            identify a local file.
    """
    del kwargs
    from pxr import Sdf  # noqa: PLC0415

    asset_path = Path(cfg.asset_path).expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"Newton MJCF file does not exist: {asset_path}")

    prim = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    prim.CreateAttribute(NEWTON_MJCF_ASSET_PATH_ATTR, Sdf.ValueTypeNames.Asset, custom=True).Set(
        Sdf.AssetPath(str(asset_path))
    )
    prim.CreateAttribute(NEWTON_MJCF_SELF_COLLISION_ATTR, Sdf.ValueTypeNames.Bool, custom=True).Set(cfg.self_collision)
    return prim
