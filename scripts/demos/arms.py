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

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

##
# Pre-defined configs
##
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # isort:skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip
from isaaclab_assets.robots.kinova import KINOVA_GEN3_N7_CFG, KINOVA_JACO2_N6S300_CFG, KINOVA_JACO2_N7S300_CFG  # isort:skip
from isaaclab_assets.robots.sawyer import SAWYER_CFG  # isort:skip
from isaaclab_assets.robots.universal_robots import UR10_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class DemoSceneCfg(InteractiveSceneCfg):
    """Configuration for the manipulator demo scene."""

    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


def design_scene() -> dict[str, "Articulation"]:
    """Designs the scene."""
    origins = (
        (-1.0, -2.0, 0.0),
        (1.0, -2.0, 0.0),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 2.0, 0.0),
        (1.0, 2.0, 0.0),
    )
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
    table_cfgs = []
    for index, (origin, (usd_path, scale, position)) in enumerate(zip(origins, table_specs, strict=True), start=1):
        table_cfgs.append(
            AssetBaseCfg(
                prim_path=f"/World/Origin{index}/Table",
                spawn=sim_utils.UsdFileCfg(usd_path=usd_path, scale=scale),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=tuple(value + offset for value, offset in zip(position, origin))
                ),
            )
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
    for cfg, origin, (_, _, table_position) in zip(robot_cfgs, origins, table_specs, strict=True):
        cfg.init_state.pos = (origin[0], origin[1], origin[2] + table_position[2])
    robot_cfgs[0].spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"

    scene_cfg = DemoSceneCfg(num_envs=1, env_spacing=0.0)
    for index, table_cfg in enumerate(table_cfgs, start=1):
        setattr(scene_cfg, f"table_{index}", table_cfg)
    for name, robot_cfg in zip(names, robot_cfgs, strict=True):
        setattr(scene_cfg, name, robot_cfg)
    scene = scene_cfg.class_type(scene_cfg)

    return scene.articulations


def run_simulator(sim: "sim_utils.SimulationContext", entities: dict[str, "Articulation"]):
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
            # reset the scene entities
            for robot in entities.values():
                # root state
                root_pose = robot.data.default_root_pose.torch.clone()
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
        scene_entities = design_scene()
        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        # Run the simulator
        run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
