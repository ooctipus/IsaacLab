# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Visual test script for the pva sensor from the Orbit framework.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaacsim import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(description="Pva Test Script")
parser.add_argument("--visualize", action="store_true", help="Open a window to display sensor output.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to clone.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default="generator",
    choices=["generator", "usd", "plane"],
    help="Type of terrain to import. Can be 'generator' or 'usd' or 'plane'.",
)
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": not args_cli.visualize}
simulation_app = SimulationApp(config)


"""Rest everything follows."""

import logging
import traceback

import torch
from isaaclab_physx.renderers.kit_viewport_utils import _set_kit_camera_view

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.cloner import CloneCfg, ReplicateSession
from isaaclab.sensors.pva import Pva, PvaCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer

# import logger
logger = logging.getLogger(__name__)


def design_scene(sim: SimulationContext, num_envs: int = 2048) -> tuple[RigidObject, Pva]:
    """Design the scene."""
    terrain_cfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type=args_cli.terrain_type,
        terrain_generator=ROUGH_TERRAINS_CFG,
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
        max_init_terrain_level=None,
        num_envs=1,
    )
    light_cfg = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=2000))
    ball_cfg = RigidObjectCfg(
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        prim_path="{ENV_REGEX_NS}/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )
    pva_cfg = PvaCfg(prim_path="{ENV_REGEX_NS}/ball", debug_vis=args_cli.visualize)
    pva_cfg.visualizer_cfg.markers["arrow"].scale = (1.0, 0.2, 0.2)
    with ReplicateSession([CloneCfg(), terrain_cfg, light_cfg, ball_cfg, pva_cfg], num_envs, 2.0):
        _ = terrain_cfg.class_type(terrain_cfg)
        light_cfg.spawn.func(light_cfg.prim_path, light_cfg.spawn)
        balls = ball_cfg.class_type(ball_cfg)
        pva = pva_cfg.class_type(pva_cfg)
    return balls, pva


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(SimulationCfg())
    # Set main camera
    _set_kit_camera_view([0.0, 30.0, 25.0], [0.0, 0.0, -2.5], "/OmniverseKit_Persp")

    # Parameters
    num_envs = args_cli.num_envs
    # Design the scene
    balls, pva = design_scene(sim=sim, num_envs=num_envs)

    # Play simulator and init the Pva
    sim.reset()

    # Print the sensor information
    print(pva)

    # Get the ball initial positions
    sim.step(render=args_cli.visualize)
    balls.update(sim.get_physics_dt())
    ball_initial_positions = balls.data.root_pos_w.torch.clone()
    ball_initial_orientations = balls.data.root_quat_w.torch.clone()

    # Create a counter for resetting the scene
    step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=args_cli.visualize)
            continue
        # Reset the scene
        if step_count % 500 == 0:
            # reset ball positions
            balls.write_root_pose_to_sim(torch.cat([ball_initial_positions, ball_initial_orientations], dim=-1))
            balls.reset()
            # reset the sensor
            pva.reset()
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update the pva sensor
        with Timer(f"Pva sensor update with {num_envs}"):
            pva.update(dt=sim.get_physics_dt(), force_recompute=True)
        # Update counter
        step_count += 1


if __name__ == "__main__":
    try:
        # Run the main function
        main()
    except Exception as err:
        logger.error(err)
        logger.error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
