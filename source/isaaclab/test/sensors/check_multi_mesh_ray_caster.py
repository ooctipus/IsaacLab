# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""
This script shows how to use the multi-mesh ray caster from the Isaac Lab framework.

.. code-block:: bash

    # Usage
    uv run python source/isaaclab/test/sensors/check_multi_mesh_ray_caster.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Ray Caster Test Script")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to clone.")
parser.add_argument("--num_objects", type=int, default=0, help="Number of additional objects to clone.")
parser.add_argument(
    "--terrain_type",
    type=str,
    default="generator",
    help="Type of terrain to import. Can be 'generator' or 'usd' or 'plane'.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import random

import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab import cloner as lab_cloner
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.timer import Timer


def main():
    """Main function."""
    num_envs = args_cli.num_envs
    balls_cfg = RigidObjectCfg(
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
    object_cfgs = [
        RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/object_{i}",
            spawn=sim_utils.CuboidCfg(
                size=(0.5 + random.random() * 0.5, 0.5 + random.random() * 0.5, 0.1 + random.random() * 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(i / args_cli.num_objects, 0.0, 1.0 - i / args_cli.num_objects)
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(random.random(), random.random(), 1.0),
                rot=tuple(quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.rand(1) * torch.pi)[0].tolist()),
            ),
        )
        for i in range(args_cli.num_objects)
    ]
    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type=args_cli.terrain_type,
        terrain_generator=ROUGH_TERRAINS_CFG,
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
        max_init_terrain_level=0,
        num_envs=1,
        env_spacing=10.0,
    )
    mesh_targets: list[MultiMeshRayCasterCfg.RaycastTargetCfg] = [
        MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/ground", track_mesh_transforms=False),
    ]
    if args_cli.num_objects != 0:
        mesh_targets.append(
            MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/object_[^/]*", track_mesh_transforms=True)
        )
    ray_caster_cfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        mesh_prim_paths=mesh_targets,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0)),
        ray_alignment="yaw",
        debug_vis=not args_cli.headless,
    )

    sim = SimulationContext(SimulationCfg())
    sim.set_camera_view([0.0, 30.0, 25.0], [0.0, 0.0, -2.5])
    light_cfg = sim_utils.DistantLightCfg(intensity=2000)
    light_cfg.func("/World/light", light_cfg)
    with lab_cloner.ReplicateSession(
        [lab_cloner.CloneCfg(), terrain_importer_cfg, balls_cfg, *object_cfgs, ray_caster_cfg], num_envs, 10.0
    ):
        _terrain = terrain_importer_cfg.class_type(terrain_importer_cfg)
        balls = balls_cfg.class_type(balls_cfg)
        _objects = [cfg.class_type(cfg) for cfg in object_cfgs]
        ray_caster = ray_caster_cfg.class_type(ray_caster_cfg)

    # Play simulator
    sim.reset()

    # Initialize the views
    # -- balls
    print(balls)
    # Print the sensor information
    print(ray_caster)

    # Get the initial positions of the balls
    ball_initial_poses = balls.data.root_pose_w.torch.clone()
    ball_initial_velocities = balls.data.root_vel_w.torch.clone()

    # Create a counter for resetting the scene
    step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=False)
            continue
        # Reset the scene
        if step_count % 500 == 0:
            # sample random indices to reset
            reset_indices = torch.randint(0, num_envs, (num_envs // 2,), device=sim.device)
            # reset the balls
            balls.write_root_pose_to_sim(ball_initial_poses[reset_indices], env_ids=reset_indices)
            balls.write_root_velocity_to_sim(ball_initial_velocities[reset_indices], env_ids=reset_indices)
            balls.reset(reset_indices)
            # reset the sensor
            ray_caster.reset(reset_indices)
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update the ray-caster
        with Timer(f"Ray-caster update with {num_envs} x {ray_caster.num_rays} rays"):
            ray_caster.update(dt=sim.get_physics_dt(), force_recompute=True)
        # Update counter
        step_count += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
