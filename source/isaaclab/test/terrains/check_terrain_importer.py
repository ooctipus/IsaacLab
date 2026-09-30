# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the terrain generator from the Isaac Lab framework.

The terrains are generated using the :class:`TerrainGenerator` class and imported using the :class:`TerrainImporter`
class. The terrains can be imported from a file or generated procedurally.

Example usage:

.. code-block:: bash

    # generate terrain
    # -- use physics sphere mesh
    uv run python source/isaaclab/test/terrains/check_terrain_importer.py --terrain_type generator
    # -- usd usd sphere geom
    uv run python source/isaaclab/test/terrains/check_terrain_importer.py --terrain_type generator --geom_sphere

    # usd terrain
    uv run python source/isaaclab/test/terrains/check_terrain_importer.py --terrain_type usd

    # plane terrain
    uv run python source/isaaclab/test/terrains/check_terrain_importer.py --terrain_type plane
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# isaaclab
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script shows how to use the terrain importer.")
parser.add_argument("--geom_sphere", action="store_true", default=False, help="Whether to use sphere mesh or shape.")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of balls to clone.")
parser.add_argument(
    "--terrain_type",
    type=str,
    choices=["generator", "usd", "plane"],
    default="generator",
    help="Type of terrain to import. Can be 'generator' or 'usd' or 'plane'.",
)
parser.add_argument(
    "--color_scheme",
    type=str,
    default="height",
    choices=["height", "random", "none"],
    help="The color scheme to use for the generated terrain.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab import cloner as lab_cloner
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

_BALL_SPAWN_CFG = (sim_utils.SphereCfg if args_cli.geom_sphere else sim_utils.MeshSphereCfg)(
    radius=0.25,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.2, dynamic_friction=1.0, restitution=0.0),
)


@configclass
class DirectCfg:
    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg())
    num_envs: int = args_cli.num_envs
    env_spacing: float = 2.0
    terrain: terrain_gen.TerrainImporterCfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type=args_cli.terrain_type,
        terrain_generator=ROUGH_TERRAINS_CFG.replace(curriculum=True, color_scheme=args_cli.color_scheme),
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
    )
    light: AssetBaseCfg = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(intensity=1000.0))
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        spawn=_BALL_SPAWN_CFG,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )


def main():
    """Generates a terrain from isaaclab."""
    cfg = DirectCfg()
    sim = SimulationContext(cfg.sim)
    # Set main camera
    sim.set_camera_view(eye=(0.0, 30.0, 25.0), target=(0.0, 0.0, -2.5))

    scene_cfgs = (cfg.terrain, cfg.light, cfg.ball)
    with lab_cloner.ReplicateSession(scene_cfgs, cfg.num_envs, cfg.env_spacing):
        terrain = cfg.terrain.class_type(cfg.terrain)
        (light_source,) = lab_cloner.query.cfg_source_paths(sim.get_clone_plan(), cfg.light)
        cfg.light.spawn.func(light_source, cfg.light.spawn)
        ball = cfg.ball.class_type(cfg.ball)

    lab_cloner.filter_collisions(
        sim.stage,
        sim.cfg.physics_prim_path,
        "/World/collisions",
        prim_paths=lab_cloner.query.env_root_paths(sim.get_clone_plan()),
        global_paths=[cfg.terrain.prim_path],
    )
    sim.reset()

    ball_initial_pose = ball.data.default_root_pose.torch.clone()
    ball_initial_pose[:, :3] = terrain.env_origins
    ball_initial_pose[:, 2] += 5.0
    ball_initial_velocity = ball.data.default_root_vel.torch.clone()

    # Create a counter for resetting the scene
    step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step()
            continue
        # Reset the scene
        if step_count % 500 == 0:
            ball.write_root_pose_to_sim_index(root_pose=ball_initial_pose)
            ball.write_root_velocity_to_sim_index(root_velocity=ball_initial_velocity)
            ball.reset()
            step_count = 0
        # Step simulation
        sim.step()
        # Update counter
        step_count += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
