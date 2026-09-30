# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to view ARL Robot 1.

.. code-block:: bash

    # Usage with default PhysX physics and no visualizer.
    uv run python scripts/demos/arl_robot_1.py

    # Usage with Newton visualizer and default PhysX physics.
    uv run python scripts/demos/arl_robot_1.py visualizer=newton_gl

"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendSimulationCfg

parser = argparse.ArgumentParser(
    description="View ARL Robot 1 with Lee Position Controller.",
    conflict_handler="resolve",
)
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import torch
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.cloner import ReplicateSession

##
# Pre-defined configs
##
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.controllers.lee_position_control_cfg import LeePosControllerCfg

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG


@configclass
class DemoCfg:
    """ARL Robot 1 demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=0.01,
        device=args_cli.device,
        physics=preset(default=PhysxCfg(), isaacsim_physx=PhysxCfg()),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.53, 0.81, 0.92)),
    )
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/Robot")
    controller: LeePosControllerCfg = LeePosControllerCfg(
        K_pos_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
        K_vel_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
        K_rot_range=((1.6, 1.6, 0.25), (1.85, 1.85, 0.4)),
        K_angvel_range=((0.4, 0.4, 0.075), (0.5, 0.5, 0.09)),
        max_inclination_angle_rad=1.0471975511965976,
        max_yaw_rate=1.0471975511965976,
    )


def main():
    """Main function to spawn arl_robot_1."""
    cfg = resolve_config(DemoCfg(), config_overrides)
    with launch_simulation(cfg, args_cli):
        # Create simulation context
        sim = sim_utils.SimulationContext(cfg.sim)

        cfg.robot.actuators["thrusters"].dt = cfg.sim.dt
        with ReplicateSession((cfg.light, cfg.ground, cfg.robot), 1, 0.0):
            cfg.light.class_type(cfg.light)
            cfg.ground.class_type(cfg.ground)
            robot = cfg.robot.class_type(cfg.robot)

        # Play the simulator
        sim.reset()

        # Create Lee position controller
        controller = cfg.controller.class_type(cfg.controller, robot, num_envs=1, device=str(sim.device))

        # Get allocation matrix and compute pseudoinverse
        allocation_matrix = torch.tensor(cfg.robot.allocation_matrix, device=sim.device, dtype=torch.float32)
        # allocation_matrix is (6, num_thrusters), we need pseudoinverse for wrench -> thrust
        alloc_pinv = torch.linalg.pinv(allocation_matrix)  # Shape: (num_thrusters, 6)

        # Position command: hover in place (zero position, zero yaw)
        pos_command = torch.zeros((1, 4), device=sim.device)  # [x, y, z, yaw]
        pos_command[0, 2] = 1.0  # Hover at 1 meter height

        # Simulation loop
        print("[INFO] Starting demo with Lee Position Controller. Press Ctrl+C to stop.")

        # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
        while sim.is_headless_or_exist_active_visualizer():
            # Compute wrench from position controller
            wrench = controller.compute(pos_command)  # Shape: (1, 6)

            # Allocate wrench to thrusters: thrust = pinv(A) @ wrench
            thrust_cmd = torch.matmul(wrench, alloc_pinv.T)  # Shape: (1, num_thrusters)
            thrust_cmd = thrust_cmd.clamp(min=0.0)  # Ensure non-negative thrust

            # Apply thrust
            robot.set_thrust_target(thrust_cmd)

            # Step simulation
            robot.write_data_to_sim()
            sim.step()

            # Update robot
            robot.update(cfg.sim.dt)


if __name__ == "__main__":
    main()
