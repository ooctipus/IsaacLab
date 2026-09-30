# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates procedural terrains with flat patches.

Example usage:

.. code-block:: bash

    # Generate terrain with height color scheme
    uv run python scripts/demos/procedural_terrain.py --color_scheme height

    # Generate terrain with random color scheme
    uv run python scripts/demos/procedural_terrain.py --color_scheme random

    # Generate terrain with no color scheme
    uv run python scripts/demos/procedural_terrain.py --color_scheme none

    # Generate terrain with curriculum
    uv run python scripts/demos/procedural_terrain.py --use_curriculum

    # Generate terrain with curriculum along with flat patches
    uv run python scripts/demos/procedural_terrain.py --use_curriculum --show_flat_patches

"""

from __future__ import annotations

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendCameraCfg, MultiBackendSimulationCfg

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates procedural terrain generation.",
    conflict_handler="resolve",
)
parser.add_argument(
    "--color_scheme",
    type=str,
    default="none",
    choices=["height", "random", "none"],
    help="Color scheme to use for the terrain generation.",
)
parser.add_argument(
    "--use_curriculum",
    action="store_true",
    default=False,
    help="Whether to use the curriculum for the terrain generation.",
)
parser.add_argument(
    "--show_flat_patches",
    action="store_true",
    default=False,
    help="Whether to show the flat patches computed during the terrain generation.",
)
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import random

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg

##
# Pre-defined configs
##
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.terrains.sub_terrain_cfg import FlatPatchSamplingCfg
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_physx.physics import PhysxCfg  # isort:skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort:skip

_TERRAIN_GENERATOR_CFG = ROUGH_TERRAINS_CFG.replace(
    curriculum=args_cli.use_curriculum, color_scheme=args_cli.color_scheme
)
if args_cli.show_flat_patches:
    for name, sub_terrain_cfg in _TERRAIN_GENERATOR_CFG.sub_terrains.items():
        sub_terrain_cfg.flat_patch_sampling = {
            name: FlatPatchSamplingCfg(num_patches=10, patch_radius=0.5, max_height_diff=0.05)
        }

_FLAT_PATCH_MARKERS = {
    name: sim_utils.CylinderCfg(
        radius=0.5,
        height=0.1,
        visual_material=sim_utils.GlassMdlCfg(glass_color=(random.random(), random.random(), random.random())),
    )
    for name in _TERRAIN_GENERATOR_CFG.sub_terrains
}


@configclass
class DemoCfg:
    """Procedural-terrain demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=0.01,
        device=args_cli.device,
        physics=preset(default=PhysxCfg(), isaacsim_physx=PhysxCfg()),
    )
    camera: MultiBackendCameraCfg = MultiBackendCameraCfg()
    num_envs: int = 2048
    env_spacing: float = 3.0
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=_TERRAIN_GENERATOR_CFG,
        debug_vis=True,
    )
    terrain.visual_material = None if args_cli.color_scheme in ("height", "random") else terrain.visual_material
    flat_patches: VisualizationMarkersCfg | None = (
        VisualizationMarkersCfg(prim_path="/Visuals/TerrainFlatPatches", markers=_FLAT_PATCH_MARKERS)
        if args_cli.show_flat_patches
        else None
    )


def main():
    """Main function."""
    cfg = resolve_config(DemoCfg(), config_overrides)
    with launch_simulation(cfg, args_cli):
        # Initialize the simulation context
        sim = sim_utils.SimulationContext(cfg.sim)
        # Set main camera
        sim.set_camera_view(eye=[15.0, 15.0, 15.0], target=[0.0, 0.0, 0.0])
        asset_cfgs = tuple(
            asset_cfg
            for asset_cfg in (cfg.light, cfg.terrain, cfg.terrain.visualizer_cfg, cfg.flat_patches, cfg.camera)
            if asset_cfg is not None
        )
        with cloner.ReplicateSession(asset_cfgs, cfg.num_envs, cfg.env_spacing):
            _camera = cfg.camera.class_type(cfg.camera) if cfg.camera is not None else None
            terrain = cfg.terrain.class_type(cfg.terrain)
            cfg.light.class_type(cfg.light)
            flat_patches = cfg.flat_patches.class_type(cfg.flat_patches) if cfg.flat_patches is not None else None
        if flat_patches is not None:
            patch_locations = [locations.view(-1, 3) for locations in terrain.flat_patches.values()]
            patch_indices = [i for i, locations in enumerate(patch_locations) for _ in range(len(locations))]
            flat_patches.visualize(torch.cat(patch_locations), marker_indices=patch_indices)
        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        while sim.is_headless_or_exist_active_visualizer():
            sim.step()


if __name__ == "__main__":
    # run the main function
    main()
