# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a pick-and-place robot equipped with a surface gripper and interact with it.

.. code-block:: bash

    # Usage
    uv run python scripts/tutorials/01_assets/run_surface_gripper.py --device=cpu

When running this script make sure the --device flag is set to cpu. This is because the surface gripper is
currently only supported on the CPU.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a Surface Gripper.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# tutorials should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import warp as wp
from isaaclab_physx.assets import SurfaceGripper, SurfaceGripperCfg

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import PICK_AND_PLACE_CFG  # isort:skip


def design_scene(sim: SimulationContext) -> tuple[dict, torch.Tensor]:
    """Designs the scene."""
    ground_cfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light_cfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot_cfg = PICK_AND_PLACE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    gripper_cfg = SurfaceGripperCfg(prim_path="{ENV_REGEX_NS}/Robot/picker_head/SurfaceGripper")
    # We can then set different parameters for the surface gripper, note that if these parameters are not set,
    # the View will try to read them from the prim.
    gripper_cfg.max_grip_distance = 0.1  # [m] (Maximum distance at which the gripper can grasp an object)
    gripper_cfg.shear_force_limit = 500.0  # [N] (Force limit in the direction perpendicular direction)
    gripper_cfg.coaxial_force_limit = 500.0  # [N] (Force limit in the direction of the gripper's axis)
    gripper_cfg.retry_interval = 0.1  # seconds (Time the gripper will stay in a grasping state)
    with cloner.ReplicateSession(
        (ground_cfg, light_cfg, robot_cfg, gripper_cfg),
        2,
        5.5,
        sim.device,
        env_template="/World/Origin{}",
    ):
        ground_cfg.spawn.func(ground_cfg.prim_path, ground_cfg.spawn)
        light_cfg.spawn.func(light_cfg.prim_path, light_cfg.spawn)
        pick_and_place_robot = robot_cfg.class_type(robot_cfg)
        surface_gripper = gripper_cfg.class_type(gripper_cfg)

    plan = sim.get_clone_plan()
    assert plan is not None and plan.positions is not None
    scene_entities = {"pick_and_place_robot": pick_and_place_robot, "surface_gripper": surface_gripper}
    return scene_entities, plan.positions


def run_simulator(
    sim: sim_utils.SimulationContext, entities: dict[str, Articulation | SurfaceGripper], origins: torch.Tensor
):
    """Runs the simulation loop."""
    # Extract scene entities
    robot: Articulation = entities["pick_and_place_robot"]
    surface_gripper: SurfaceGripper = entities["surface_gripper"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_pose = robot.data.default_root_pose.torch.clone()
            root_pose[:, :3] += origins
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            root_vel = robot.data.default_root_vel.torch.clone()
            robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
            # set joint positions with some noise
            joint_pos, joint_vel = (
                robot.data.default_joint_pos.torch.clone(),
                robot.data.default_joint_vel.torch.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
            # Opens the gripper and makes sure the gripper is in the open state
            surface_gripper.reset()
            print("[INFO]: Resetting gripper state...")

        # Sample a random command between -1 and 1.
        gripper_commands = torch.rand(surface_gripper.num_instances) * 2.0 - 1.0
        # The gripper behavior is as follows:
        # -1 < command < -0.3 --> Gripper is Opening
        # -0.3 < command < 0.3 --> Gripper is Idle
        # 0.3 < command < 1 --> Gripper is Closing
        print(f"[INFO]: Gripper commands: {gripper_commands}")
        mapped_commands = [
            "Opening" if command < -0.3 else "Closing" if command > 0.3 else "Idle" for command in gripper_commands
        ]
        print(f"[INFO]: Mapped commands: {mapped_commands}")
        # Set the gripper command
        surface_gripper.set_grippers_command(gripper_commands)
        # Write data to sim
        surface_gripper.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Read the gripper state from the simulation
        surface_gripper.update(sim_dt)
        # Read the gripper state from the buffer
        surface_gripper_state = surface_gripper.state
        # The gripper state is a list of integers that can be mapped to the following:
        # -1 --> Open
        # 0 --> Closing
        # 1 --> Closed
        # Print the gripper state
        print(f"[INFO]: Gripper state: {surface_gripper_state}")
        mapped_commands = [
            "Open" if state == -1 else "Closing" if state == 0 else "Closed"
            for state in wp.to_torch(surface_gripper_state).tolist()
        ]
        print(f"[INFO]: Mapped commands: {mapped_commands}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.75, 7.5, 10.0], [2.75, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins = design_scene(sim)
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
