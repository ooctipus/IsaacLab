# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage with default PhysX physics and default kit visualizer.
    uv run python scripts/demos/arms.py

    # Usage with Newton visualizer and default PhysX physics.
    uv run python scripts/demos/arms.py --visualizer newton

    # Usage with Newton (MJWarp) physics and default kit visualizer.
    uv run python scripts/demos/arms.py --physics newton_mjwarp

    # Usage with Newton visualizer and Newton (MJWarp) physics.
    uv run python scripts/demos/arms.py --visualizer newton --physics newton_mjwarp

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="This script demonstrates different single-arm manipulators.",
    conflict_handler="resolve",
)
parser.add_argument(
    "--physics", default="isaacsim_physx", choices=["isaacsim_physx", "newton_mjwarp"], help="Physics backend."
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.assets import AssetBaseCfg

##
# Pre-defined configs
##
from isaaclab.physics import PhysicsCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # isort:skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip
from isaaclab_assets.robots.kinova import KINOVA_GEN3_N7_CFG, KINOVA_JACO2_N6S300_CFG, KINOVA_JACO2_N7S300_CFG  # isort:skip
from isaaclab_assets.robots.sawyer import SAWYER_CFG  # isort:skip
from isaaclab_assets.robots.universal_robots import UR10_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene(sim: "sim_utils.SimulationContext") -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    ground_cfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light_cfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    origins = define_origins(num_origins=6, spacing=2.0)
    seattle_table = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    stand = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd"
    thorlabs_table = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"
    table_specs = (
        (seattle_table, None, (0.55, 0.0, 1.05)),
        (stand, (2.0, 2.0, 2.0), (0.0, 0.0, 1.03)),
        (thorlabs_table, None, (0.0, 0.0, 0.8)),
        (thorlabs_table, None, (0.0, 0.0, 0.8)),
        (seattle_table, None, (0.55, 0.0, 1.05)),
        (stand, (2.0, 2.0, 2.0), (0.0, 0.0, 1.03)),
    )
    table_cfgs = tuple(
        AssetBaseCfg(
            prim_path=f"/World/Origin{index}/Table",
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path, scale=scale),
            init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        )
        for index, (usd_path, scale, position) in enumerate(table_specs, start=1)
    )
    names = ("franka_panda", "ur10", "kinova_j2n7s300", "kinova_j2n6s300", "kinova_gen3n7", "sawyer")
    templates = (
        FRANKA_PANDA_CFG,
        UR10_CFG,
        KINOVA_JACO2_N7S300_CFG,
        KINOVA_JACO2_N6S300_CFG,
        KINOVA_GEN3_N7_CFG,
        SAWYER_CFG,
    )
    robot_cfgs = tuple(cfg.replace(prim_path=f"/World/Origin{index}/Robot") for index, cfg in enumerate(templates, 1))
    for cfg, (_, _, table_position) in zip(robot_cfgs, table_specs, strict=True):
        cfg.init_state.pos = (0.0, 0.0, table_position[2])
    robot_cfgs[0].spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"

    with cloner.ReplicateSession((ground_cfg, light_cfg, *table_cfgs, *robot_cfgs), 1, 0.0, sim.device):
        ground_cfg.spawn.func(ground_cfg.prim_path, ground_cfg.spawn)
        light_cfg.spawn.func(light_cfg.prim_path, light_cfg.spawn)
        for index, (origin, table_cfg) in enumerate(zip(origins, table_cfgs, strict=True), start=1):
            sim_utils.create_prim(f"/World/Origin{index}", "Xform", translation=origin)
            table_cfg.spawn.func(table_cfg.prim_path, table_cfg.spawn, translation=table_cfg.init_state.pos)
        robots = [cfg.class_type(cfg) for cfg in robot_cfgs]
    scene_entities = dict(zip(names, robots, strict=True))
    return scene_entities, origins


def run_simulator(sim: "sim_utils.SimulationContext", entities: dict[str, "Articulation"], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
    while sim.is_headless_or_exist_active_visualizer():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset the scene entities
            for index, robot in enumerate(entities.values()):
                # root state
                root_pose = robot.data.default_root_pose.torch.clone()
                root_pose[:, :3] += origins[index]
                robot.write_root_pose_to_sim_index(root_pose=root_pose)
                root_vel = robot.data.default_root_vel.torch.clone()
                robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
                # set joint positions
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos.torch.clone(),
                    robot.data.default_joint_vel.torch.clone(),
                )
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                # clear internal buffers
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply random actions to the robots
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = robot.data.default_joint_pos.torch + torch.randn_like(robot.data.joint_pos.torch) * 0.1
            soft_limits = robot.data.soft_joint_pos_limits.torch
            joint_pos_target = joint_pos_target.clamp_(soft_limits[..., 0], soft_limits[..., 1])
            # apply action to the robot
            robot.set_joint_position_target_index(target=joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # The default newton mjwarp solver configuration needs to be tuned for these arms.
        if isinstance(physics_cfg, NewtonCfg) and isinstance(physics_cfg.solver_cfg, MJWarpSolverCfg):
            physics_cfg.solver_cfg.njmax = 70
            physics_cfg.solver_cfg.nconmax = 70
            physics_cfg.solver_cfg.ls_iterations = 40
            physics_cfg.solver_cfg.cone = "elliptic"
            physics_cfg.solver_cfg.impratio = 100
            physics_cfg.solver_cfg.ls_parallel = False
            physics_cfg.solver_cfg.integrator = "implicitfast"
            physics_cfg.num_substeps = 2

        # Initialize the simulation context
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
        # design scene
        scene_entities, scene_origins = design_scene(sim)
        scene_origins = torch.tensor(scene_origins, device=sim.device)
        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        # Run the simulator
        run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
