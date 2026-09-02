# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates different legged robots.

.. code-block:: bash

    # Usage with default PhysX physics and default kit visualizer.
    uv run python scripts/demos/quadrupeds.py

    # Usage with Newton visualizer and default PhysX physics.
    uv run python scripts/demos/quadrupeds.py --visualizer newton

    # Usage with Newton (MJWarp) physics and default kit visualizer.
    uv run python scripts/demos/quadrupeds.py --physics newton_mjwarp

    # Usage with Newton visualizer and Newton (MJWarp) physics.
    uv run python scripts/demos/quadrupeds.py --visualizer newton --physics newton_mjwarp

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="This script demonstrates different legged robots.",
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
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG  # isort:skip
from isaaclab_assets.robots.spot import SPOT_CFG  # isort:skip
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class DemoSceneCfg(InteractiveSceneCfg):
    """Configuration for the quadruped demo scene."""

    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )


def design_scene() -> dict[str, "Articulation"]:
    """Designs the scene."""
    origins = (
        (-1.875, -0.625, 0.0),
        (-0.625, -0.625, 0.0),
        (0.625, -0.625, 0.0),
        (1.875, -0.625, 0.0),
        (-1.875, 0.625, 0.0),
        (-0.625, 0.625, 0.0),
        (0.625, 0.625, 0.0),
    )
    names = ("anymal_b", "anymal_c", "anymal_d", "unitree_a1", "unitree_go1", "unitree_go2", "spot")
    templates = (ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG, UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG, SPOT_CFG)
    robot_cfgs = tuple(cfg.replace(prim_path=f"/World/Origin{index}/Robot") for index, cfg in enumerate(templates, 1))
    scene_cfg = DemoSceneCfg(num_envs=1, env_spacing=0.0)
    for name, origin, robot_cfg in zip(names, origins, robot_cfgs, strict=True):
        robot_cfg.init_state.pos = tuple(value + offset for value, offset in zip(robot_cfg.init_state.pos, origin))
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
        # Reset robots every 200 steps.
        if count % 200 == 0:
            # reset counters
            count = 0
            # reset robots
            for robot in entities.values():
                # root state
                root_pose = robot.data.default_root_pose.torch.clone()
                robot.write_root_pose_to_sim_index(root_pose=root_pose)
                root_vel = robot.data.default_root_vel.torch.clone()
                robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
                # joint state
                joint_pos = robot.data.default_joint_pos.torch.clone()
                robot.write_joint_position_to_sim_index(position=joint_pos)
                joint_vel = robot.data.default_joint_vel.torch.clone()
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                # reset the internal state
                robot.reset()
            print("[INFO]: Reset robots' state...")
        # Apply default actions to the quadrupedal robots.
        for robot in entities.values():
            # generate random joint positions
            joint_pos_target = robot.data.default_joint_pos.torch + torch.randn_like(robot.data.joint_pos.torch) * 0.1
            # apply action to the robot
            robot.set_joint_position_target_index(target=joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update counter
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        dt = 1 / 200
        sim_cfg: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=dt, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
        scene_entities = design_scene()
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene_entities)


if __name__ == "__main__":
    main()
