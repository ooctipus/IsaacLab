# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the ray-caster sensor.

.. code-block:: bash

    # Usage
    uv run python scripts/tutorials/04_sensors/run_ray_caster.py --viz kit

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Ray Caster Test Script")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer


def design_scene(sim: sim_utils.SimulationContext) -> dict:
    """Design the scene."""
    sensor_cfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        spawn=None,
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.0)),
        ray_alignment="yaw",
        debug_vis=not args_cli.headless,
    )
    ground_cfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd"),
    )
    light_cfg = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=2000))
    ball_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )
    with cloner.ReplicateSession(
        (ground_cfg, light_cfg, ball_cfg, sensor_cfg),
        4,
        0.5,
        sim.device,
        env_template="/World/Origin{}",
    ):
        ground_cfg.spawn.func(ground_cfg.prim_path, ground_cfg.spawn)
        light_cfg.spawn.func(light_cfg.prim_path, light_cfg.spawn)
        balls = ball_cfg.class_type(ball_cfg)
        ray_caster = sensor_cfg.class_type(sensor_cfg)

    # return the scene information
    scene_entities = {"balls": balls, "ray_caster": ray_caster}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # Extract scene_entities for simplified notation
    ray_caster: RayCaster = scene_entities["ray_caster"]
    balls: RigidObject = scene_entities["balls"]

    # define an initial position of the sensor
    ball_default_pose = balls.data.default_root_pose.torch.clone()
    ball_default_pose[:, :3] = torch.rand_like(ball_default_pose[:, :3]) * 10
    ball_default_vel = balls.data.default_root_vel.torch.clone()

    # Create a counter for resetting the scene
    step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # Reset the scene
        if step_count % 250 == 0:
            # reset the balls
            balls.write_root_pose_to_sim_index(root_pose=ball_default_pose)
            balls.write_root_velocity_to_sim_index(root_velocity=ball_default_vel)
            # reset the sensor
            ray_caster.reset()
            # reset the counter
            step_count = 0
        # Step simulation
        sim.step()
        # Update the ray-caster
        with Timer(
            f"Ray-caster update with {4} x {ray_caster.num_rays} rays with max height of"
            f" {torch.max(ray_caster.data.pos_w.torch).item():.2f}"
        ):
            ray_caster.update(dt=sim.get_physics_dt(), force_recompute=True)
        # Update counter
        step_count += 1


def main():
    """Main function."""
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([0.0, 15.0, 15.0], [0.0, 0.0, -2.5])
    # Design scene
    scene_entities = design_scene(sim)
    # Play simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run simulator
    run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
