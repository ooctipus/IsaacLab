# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a pick-and-place robot equipped with a surface gripper and interact with it.

.. code-block:: bash

    # Usage
    uv run python scripts/tutorials/01_assets/run_surface_gripper.py --device=cpu visualizer=kit

When running this script make sure the --device flag is set to cpu. This is because the surface gripper is
currently only supported on the CPU.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_physx.assets import SurfaceGripperCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendSimulationCfg

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a Surface Gripper.")
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import torch
import warp as wp

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.cloner import ReplicateSession
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab_physx.assets import SurfaceGripper

    from isaaclab.assets import Articulation

##
# Pre-defined configs
##
from isaaclab_assets import PICK_AND_PLACE_CFG  # isort:skip


@configclass
class TutorialCfg:
    """Surface-gripper tutorial configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(device=args_cli.device, physics=PhysxCfg())
    num_envs: int = 2
    env_spacing: float = 5.5
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = PICK_AND_PLACE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    surface_gripper: SurfaceGripperCfg = SurfaceGripperCfg(
        prim_path="{ENV_REGEX_NS}/Robot/picker_head/SurfaceGripper",
        max_grip_distance=0.1,
        shear_force_limit=500.0,
        coaxial_force_limit=500.0,
        retry_interval=0.1,
    )


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
    while sim.is_headless_or_exist_active_visualizer():
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
    cfg = resolve_config(TutorialCfg(), config_overrides)
    with launch_simulation(cfg, args_cli):
        sim = sim_utils.SimulationContext(cfg.sim)
        sim.set_camera_view([2.75, 7.5, 10.0], [2.75, 0.0, 0.0])
        with ReplicateSession((cfg.ground, cfg.light, cfg.robot, cfg.surface_gripper), cfg.num_envs, cfg.env_spacing):
            cfg.ground.class_type(cfg.ground)
            cfg.light.class_type(cfg.light)
            scene_entities = {
                "pick_and_place_robot": cfg.robot.class_type(cfg.robot),
                "surface_gripper": cfg.surface_gripper.class_type(cfg.surface_gripper),
            }
        scene_origins = sim.get_clone_plan().positions
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
