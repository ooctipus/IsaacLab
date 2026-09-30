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
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors.pva import PvaCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.timer import Timer

# import logger
logger = logging.getLogger(__name__)


@configclass
class DirectCfg:
    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg())
    num_envs: int = args_cli.num_envs
    env_spacing: float = 2.0
    terrain: terrain_gen.TerrainImporterCfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type=args_cli.terrain_type,
        terrain_generator=ROUGH_TERRAINS_CFG,
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
        max_init_terrain_level=None,
    )
    light: AssetBaseCfg = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=2000))
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )
    pva: PvaCfg = PvaCfg(prim_path="{ENV_REGEX_NS}/ball", debug_vis=args_cli.visualize)
    pva.visualizer_cfg.markers["arrow"].scale = (1.0, 0.2, 0.2)


def main():
    """Main function."""
    cfg = DirectCfg()
    sim = SimulationContext(cfg.sim)
    # Set main camera
    sim.set_camera_view([0.0, 30.0, 25.0], [0.0, 0.0, -2.5])
    with cloner.ReplicateSession(
        (cfg.terrain, cfg.light, cfg.ball, cfg.pva),
        num_clones=cfg.num_envs,
        env_spacing=cfg.env_spacing,
    ):
        cfg.terrain.class_type(cfg.terrain)
        (light_source,) = cloner.query.cfg_source_paths(sim.get_clone_plan(), cfg.light)
        cfg.light.spawn.func(light_source, cfg.light.spawn)
        balls = cfg.ball.class_type(cfg.ball)
        pva = cfg.pva.class_type(cfg.pva)

    cloner.filter_collisions(
        sim.stage,
        sim.cfg.physics_prim_path,
        "/World/collisions",
        cloner.query.env_root_paths(sim.get_clone_plan()),
        global_paths=[cfg.terrain.prim_path],
    )

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
            balls.write_root_pose_to_sim_index(
                root_pose=torch.cat([ball_initial_positions, ball_initial_orientations], dim=-1)
            )
            balls.reset()
            # reset the sensor
            pva.reset()
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update the pva sensor
        with Timer(f"Pva sensor update with {cfg.num_envs}"):
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
