# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the ray-caster sensor.

.. code-block:: bash

    uv run python scripts/tutorials/04_sensors/run_ray_caster.py visualizer=kit

"""

import argparse

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendSimulationCfg

# add argparse arguments
parser = argparse.ArgumentParser(description="Ray Caster Test Script")
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.cloner import ReplicateSession
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.timer import Timer


@configclass
class TutorialCfg:
    """Ray-caster tutorial configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(device=args_cli.device, physics=PhysxCfg())
    num_envs: int = 4
    env_spacing: float = 0.5
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd"),
    )
    light: AssetBaseCfg = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=2000))
    balls: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )
    ray_caster: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.0)),
        ray_alignment="yaw",
        debug_vis=True,
    )


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
    while sim.is_headless_or_exist_active_visualizer():
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
            f"Ray-caster update with {ray_caster.num_instances} x {ray_caster.num_rays} rays with max height of"
            f" {torch.max(ray_caster.data.pos_w.torch).item():.2f}"
        ):
            ray_caster.update(dt=sim.get_physics_dt(), force_recompute=True)
        # Update counter
        step_count += 1


def main():
    """Main function."""
    cfg = resolve_config(TutorialCfg(), config_overrides)
    with launch_simulation(cfg, args_cli):
        sim = sim_utils.SimulationContext(cfg.sim)
        sim.set_camera_view([0.0, 15.0, 15.0], [0.0, 0.0, -2.5])
        with ReplicateSession(
            (cfg.ground, cfg.light, cfg.balls, cfg.ray_caster, cfg.ray_caster.visualizer_cfg),
            cfg.num_envs,
            cfg.env_spacing,
        ):
            cfg.ground.class_type(cfg.ground)
            cfg.light.class_type(cfg.light)
            scene_entities = {
                "balls": cfg.balls.class_type(cfg.balls),
                "ray_caster": cfg.ray_caster.class_type(cfg.ray_caster),
            }
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":
    main()
