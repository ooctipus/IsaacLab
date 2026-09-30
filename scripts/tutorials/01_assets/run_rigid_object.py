# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a rigid object and interact with it.

.. code-block:: bash

    # Usage
    uv run python scripts/tutorials/01_assets/run_rigid_object.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


@configclass
class RigidObjectTutorialCfg:
    """Complete declarative input for the direct clone lifecycle."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        physics=PhysxCfg(),
    )
    num_envs = 4
    env_spacing = 0.5
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    cone: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    )


def design_scene(sim: SimulationContext, cfg: RigidObjectTutorialCfg) -> tuple[dict[str, RigidObject], torch.Tensor]:
    """Designs the scene."""
    with cloner.ReplicateSession(
        (cfg.ground, cfg.light, cfg.cone), num_clones=cfg.num_envs, env_spacing=cfg.env_spacing
    ):
        cfg.ground.class_type(cfg.ground)
        cfg.light.class_type(cfg.light)
        cone = cfg.cone.class_type(cfg.cone)
    return {"cone": cone}, sim.get_clone_plan().positions


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cone_object = entities["cone"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            # reset counters
            count = 0
            # reset root state
            root_pose = cone_object.data.default_root_pose.torch.clone()
            # sample a random position on a cylinder around the origins
            root_pose[:, :3] += origins
            root_pose[:, :3] += math_utils.sample_cylinder(
                radius=0.1, h_range=(0.25, 0.5), size=cone_object.num_instances, device=cone_object.device
            )
            # write root state to simulation
            cone_object.write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = cone_object.data.default_root_vel.torch.clone()
            cone_object.write_root_velocity_to_sim_index(root_velocity=root_vel)
            # reset buffers
            cone_object.reset()
            print("----------------------------------------")
            print("[INFO]: Resetting object state...")
        # apply sim data
        cone_object.write_data_to_sim()
        # perform step
        sim.step()
        count += 1
        # update buffers
        cone_object.update(sim_dt)
        # print the root position
        if count % 50 == 0:
            print(f"Root position (in world): {cone_object.data.root_pos_w.torch}")


def main():
    """Main function."""
    cfg = RigidObjectTutorialCfg()
    cfg.sim.device = args_cli.device
    sim = SimulationContext(cfg.sim)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins = design_scene(sim, cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
