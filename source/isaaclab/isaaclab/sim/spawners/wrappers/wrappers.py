# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pxr import Usd

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import UsdFileCfg

if TYPE_CHECKING:
    from . import wrappers_cfg


def spawn_multi_asset(
    prim_path: str | Sequence[str | None],
    cfg: wrappers_cfg.MultiAssetSpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    """Spawn multiple assets into numbered or explicitly supplied prim paths.

    Assets are created in the order they appear in ``cfg.assets_cfg``. A string path derives
    numbered siblings from its segment wildcard; a sequence supplies one exact path per variant,
    with ``None`` retaining an inactive variant's slot without spawning it.

    Args:
        prim_path: Wildcard path or exact per-variant paths to spawn the assets at.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets in (x, y, z, w) order. Default is None.
        clone_in_fabric: Whether to clone in fabric. Default is False.
        replicate_physics: Whether to replicate physics. Default is False.

    Returns:
        The created prim at the first prim path.
    """
    if not isinstance(prim_path, str):
        if len(prim_path) != len(cfg.assets_cfg):
            raise ValueError(
                f"Expected one prim path per asset configuration, got {len(prim_path)} and {len(cfg.assets_cfg)}."
            )
        asset_prim_paths = prim_path
    else:
        # split on separators only: a segment wildcard is written as a character class whose
        # text contains a '/' that is not a separator.
        split_path = sim_utils.split_path_expr(prim_path)
        prefix_path, base_name = "/".join(split_path[:-1]), split_path[-1]
        # the base name carries the index slot as a segment wildcard, in any of its spellings.
        # Normalizing to glob collapses them to the single '*' that the index replaces.
        base_glob = sim_utils.path_expr_to_glob(base_name)
        if "*" not in base_glob:
            raise ValueError(
                f" The base name '{base_name}' in the prim path '{prim_path}' must contain a segment wildcard"
                " (e.g. '.*' or '[^/]*') to indicate the path each individual multiple-asset to be spawned."
            )
        asset_prim_paths = [f"{prefix_path}/{base_glob.replace('*', str(i))}" for i in range(len(cfg.assets_cfg))]

    spawned_prim_paths: list[str] = []
    for asset_prim_path, asset_cfg in zip(asset_prim_paths, cfg.assets_cfg):
        if asset_prim_path is None:
            continue
        # append semantic tags if specified
        if cfg.semantic_tags is not None:
            if asset_cfg.semantic_tags is None:
                asset_cfg.semantic_tags = cfg.semantic_tags
            else:
                asset_cfg.semantic_tags += cfg.semantic_tags
        # override settings for properties
        attr_names = ["mass_props", "rigid_props", "collision_props", "activate_contact_sensors", "deformable_props"]
        for attr_name in attr_names:
            attr_value = getattr(cfg, attr_name)
            if hasattr(asset_cfg, attr_name) and attr_value is not None:
                setattr(asset_cfg, attr_name, attr_value)

        asset_cfg.func(
            asset_prim_path,
            asset_cfg,
            translation=translation,
            orientation=orientation,
            clone_in_fabric=clone_in_fabric,
            replicate_physics=replicate_physics,
        )
        spawned_prim_paths.append(asset_prim_path)
    if not spawned_prim_paths:
        raise ValueError("No assets were spawned. At least one spawn path must be active.")
    return sim_utils.find_first_matching_prim(spawned_prim_paths[0])


def spawn_multi_usd_file(
    prim_path: str | Sequence[str | None],
    cfg: wrappers_cfg.MultiUsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    """Spawn multiple USD files based on the provided configurations.

    This function creates configuration instances corresponding the individual USD files and
    calls the :meth:`spawn_multi_asset` method to spawn them into the scene.

    Args:
        prim_path: Wildcard path or exact per-variant paths to spawn the assets at.
        cfg: The configuration for spawning the assets.
        translation: The translation of the spawned assets. Default is None.
        orientation: The orientation of the spawned assets in (x, y, z, w) order. Default is None.
        clone_in_fabric: Whether to clone in fabric. Default is False.
        replicate_physics: Whether to replicate physics. Default is False.

    Returns:
        The created prim at the first prim path.
    """
    # needed here to avoid circular imports
    from .wrappers_cfg import MultiAssetSpawnerCfg

    # parse all the usd files
    if isinstance(cfg.usd_path, str):
        usd_paths = [cfg.usd_path]
    else:
        usd_paths = cfg.usd_path

    # make a template usd config
    usd_template_cfg = UsdFileCfg()
    for attr_name, attr_value in cfg.__dict__.items():
        # skip names we know are not present
        if attr_name in ["func", "usd_path"]:
            continue
        # set the attribute into the template
        setattr(usd_template_cfg, attr_name, attr_value)

    # create multi asset configuration of USD files
    multi_asset_cfg = MultiAssetSpawnerCfg(assets_cfg=[])
    for usd_path in usd_paths:
        usd_cfg = usd_template_cfg.replace(usd_path=usd_path)
        multi_asset_cfg.assets_cfg.append(usd_cfg)

    # propagate the contact sensor settings
    # note: the default value for activate_contact_sensors in MultiAssetSpawnerCfg is False.
    #  This ends up overwriting the usd-template-cfg's value when the `spawn_multi_asset`
    #  function is called. We hard-code the value to the usd-template-cfg's value to ensure
    #  that the contact sensor settings are propagated correctly.
    if hasattr(cfg, "activate_contact_sensors"):
        multi_asset_cfg.activate_contact_sensors = cfg.activate_contact_sensors

    # call the original function
    return spawn_multi_asset(prim_path, multi_asset_cfg, translation, orientation, clone_in_fabric, replicate_physics)
