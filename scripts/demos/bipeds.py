# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to simulate bipedal robots.

.. code-block:: bash

    # Usage with default PhysX physics and no visualizer.
    uv run python scripts/demos/bipeds.py

    # Usage with Newton visualizer and default PhysX physics.
    uv run python scripts/demos/bipeds.py visualizer=newton_gl

    # Usage with Newton (MJWarp) physics and no visualizer.
    uv run python scripts/demos/bipeds.py physics=newton_mjwarp

    # Usage with Newton visualizer and Newton (MJWarp) physics.
    uv run python scripts/demos/bipeds.py visualizer=newton_gl physics=newton_mjwarp

"""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendCameraCfg, MultiBackendSimulationCfg

parser = argparse.ArgumentParser(
    description="This script demonstrates how to simulate bipedal robots.",
    conflict_handler="resolve",
)
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.cloner import ReplicateSession

##
# Pre-defined configs
##
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics import MJWarpSolverCfg  # isort:skip
from isaaclab_physx.physics import PhysxCfg  # isort:skip
from isaaclab_assets.robots.cassie import CASSIE_CFG  # isort:skip
from isaaclab_assets.robots.unitree import G1_CFG, H1_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class DemoCfg:
    """Biped demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=0.005,
        device=args_cli.device,
        physics=preset(
            default=PhysxCfg(),
            isaacsim_physx=PhysxCfg(),
            newton_mjwarp=MJWarpSolverCfg(
                njmax=70,
                nconmax=70,
                ls_iterations=40,
                cone="elliptic",
                impratio=100,
                integrator="implicitfast",
                num_substeps=2,
            ),
        ),
    )
    camera: MultiBackendCameraCfg = MultiBackendCameraCfg()
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    cassie: ArticulationCfg = CASSIE_CFG.replace(prim_path="/World/Cassie")
    cassie.init_state.pos = (0.0, -1.0, cassie.init_state.pos[2])
    h1: ArticulationCfg = H1_CFG.replace(prim_path="/World/H1")
    g1: ArticulationCfg = G1_CFG.replace(prim_path="/World/G1")
    g1.init_state.pos = (0.0, 1.0, g1.init_state.pos[2])


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
        # update sim-time
        count += 1
        # update buffers
        for robot in robots:
            robot.update(sim_dt)


def main():
    """Main function."""
    cfg = resolve_config(DemoCfg(), config_overrides)
    with launch_simulation(cfg, args_cli):
        # Load kit helper
        sim = sim_utils.SimulationContext(cfg.sim)
        # Set main camera
        sim.set_camera_view(eye=[3.0, 0.0, 2.25], target=[0.0, 0.0, 1.0])

        asset_cfgs = tuple(
            asset_cfg
            for asset_cfg in (cfg.ground, cfg.light, cfg.cassie, cfg.h1, cfg.g1, cfg.camera)
            if asset_cfg is not None
        )
        with ReplicateSession(asset_cfgs, 1, 0.0):
            _camera = cfg.camera.class_type(cfg.camera) if cfg.camera is not None else None
            cfg.ground.class_type(cfg.ground)
            cfg.light.class_type(cfg.light)
            robots = [robot_cfg.class_type(robot_cfg) for robot_cfg in (cfg.cassie, cfg.h1, cfg.g1)]

        # Play the simulator
        sim.reset()

        # Now we are ready!
        print("[INFO]: Setup complete...")

        # Run the simulator
        run_simulator(sim, robots)


if __name__ == "__main__":
    # run the main function
    main()
