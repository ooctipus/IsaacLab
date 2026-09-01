# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to simulate bipedal robots.

.. code-block:: bash

    # Usage with default PhysX physics and default kit visualizer.
    uv run python scripts/demos/bipeds.py

    # Usage with Newton visualizer and default PhysX physics.
    uv run python scripts/demos/bipeds.py --visualizer newton

    # Usage with Newton (MJWarp) physics and default kit visualizer.
    uv run python scripts/demos/bipeds.py --physics newton_mjwarp

    # Usage with Newton visualizer and Newton (MJWarp) physics.
    uv run python scripts/demos/bipeds.py --visualizer newton --physics newton_mjwarp

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="This script demonstrates how to simulate bipedal robots.",
    conflict_handler="resolve",
)
parser.add_argument(
    "--physics", default="isaacsim_physx", choices=["isaacsim_physx", "newton_mjwarp"], help="Physics backend."
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

##
# Pre-defined configs
##
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # isort:skip
from isaaclab_assets.robots.cassie import CASSIE_CFG  # isort:skip
from isaaclab_assets.robots.unitree import G1_CFG, H1_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class DemoSceneCfg(InteractiveSceneCfg):
    """Configuration for the biped demo scene."""

    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


def design_scene() -> list["Articulation"]:
    """Designs the scene."""
    origins = [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    names = ("cassie", "h1", "g1")
    robot_cfgs = (
        CASSIE_CFG.replace(prim_path="/World/Cassie"),
        H1_CFG.replace(prim_path="/World/H1"),
        G1_CFG.replace(prim_path="/World/G1"),
    )
    scene_cfg = DemoSceneCfg(num_envs=1, env_spacing=0.0)
    for name, origin, robot_cfg in zip(names, origins, robot_cfgs, strict=True):
        robot_cfg.init_state.pos = tuple(value + offset for value, offset in zip(robot_cfg.init_state.pos, origin))
        setattr(scene_cfg, name, robot_cfg)
    scene = scene_cfg.class_type(scene_cfg)

    return list(scene.articulations.values())


def run_simulator(sim: "sim_utils.SimulationContext", robots: list["Articulation"]):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
    while sim.is_headless_or_exist_active_visualizer():
        # reset
        if count % 200 == 0:
            # reset counters
            count = 0
            for robot in robots:
                # reset dof state
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos.torch,
                    robot.data.default_joint_vel.torch,
                )
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                root_pose = robot.data.default_root_pose.torch.clone()
                robot.write_root_pose_to_sim_index(root_pose=root_pose)
                root_vel = robot.data.default_root_vel.torch.clone()
                robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
                robot.reset()
            # reset command
            print(">>>>>>>> Reset!")
        # apply action to the robot
        for robot in robots:
            robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch.clone())
            robot.write_data_to_sim()
        # perform step
        sim.step()
        count += 1
        # update buffers
        for robot in robots:
            robot.update(sim_dt)


def main():
    """Main function."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # The default newton mjwarp solver configuration needs to be tuned for these bipeds.
        if isinstance(physics_cfg, NewtonCfg) and isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg):
            physics_cfg.solver_cfg.njmax = 70
            physics_cfg.solver_cfg.nconmax = 70
            physics_cfg.solver_cfg.ls_iterations = 40
            physics_cfg.solver_cfg.cone = "elliptic"
            physics_cfg.solver_cfg.impratio = 100
            physics_cfg.solver_cfg.ls_parallel = False
            physics_cfg.solver_cfg.integrator = "implicitfast"
            physics_cfg.num_substeps = 2
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 1.0])

        # design scene
        robots = design_scene()

        # Play the simulator
        sim.reset()

        # Now we are ready!
        print("[INFO]: Setup complete...")

        # Run the simulator
        run_simulator(sim, robots)


if __name__ == "__main__":
    # run the main function
    main()
