# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the contact sensor sensor in Isaac Lab.

.. code-block:: bash

    uv run python source/isaaclab/test/sensors/test_contact_sensor.py --num_robots 2
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Contact Sensor Test Script")
parser.add_argument("--num_robots", type=int, default=128, help="Number of robots to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import CloneCfg, ReplicateSession
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.timer import Timer

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


"""
Main
"""


def main():
    """Spawns the ANYmal robot and clones it using Isaac Sim Cloner API."""

    # Load kit helper
    sim = SimulationContext(SimulationCfg(dt=0.005))
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Enable hydra scene-graph instancing
    # this is needed to visualize the scene when flatcache is enabled
    sim.set_setting("/persistent/omnihydra/useSceneGraphInstancing", True)

    num_envs = args_cli.num_robots
    ground_cfg = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(), collision_group=-1
    )
    light_cfg = AssetBaseCfg(
        prim_path="/World/Light/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.5, 3.5, 10.0)),
    )
    robot_cfg = ANYMAL_C_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ANYMAL_C_CFG.spawn.replace(activate_contact_sensors=True),
    )
    contact_sensor_cfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*_FOOT",
        track_air_time=True,
        track_contact_points=True,
        track_friction_forces=True,
        debug_vis=False,  # not args_cli.headless,
        filter_prim_paths_expr=["/World/defaultGroundPlane/GroundPlane/CollisionPlane"],
    )
    with ReplicateSession([CloneCfg(), ground_cfg, light_cfg, robot_cfg, contact_sensor_cfg], num_envs, 2.0):
        ground_cfg.spawn.func(ground_cfg.prim_path, ground_cfg.spawn)
        light_cfg.spawn.func(light_cfg.prim_path, light_cfg.spawn, translation=light_cfg.init_state.pos)
        robot = robot_cfg.class_type(robot_cfg)
        contact_sensor = contact_sensor_cfg.class_type(contact_sensor_cfg)
    # Play the simulator
    sim.reset()
    # print info
    print(contact_sensor)

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    decimation = 4
    physics_dt = sim.get_physics_dt()
    sim_dt = decimation * physics_dt
    sim_time = 0.0
    count = 0
    dt = []
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=False)
            continue
        # reset
        if count % 1000 == 0 and count != 0:
            # reset counters
            sim_time = 0.0
            count = 0
            print("=" * 80)
            print("avg dt real-time", sum(dt) / len(dt))
            print("=" * 80)

            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            dt = []

        # perform 4 steps
        for _ in range(decimation):
            # apply actions
            robot.set_joint_position_target(robot.data.default_joint_pos)
            # write commands to sim
            robot.write_data_to_sim()
            # perform step
            sim.step()
            # fetch data
            robot.update(physics_dt)
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update the buffers
        if sim.is_playing():
            with Timer() as timer:
                contact_sensor.update(sim_dt, force_recompute=True)
                dt.append(timer.time_elapsed)

            contact_sensor.update(sim_dt, force_recompute=True)
            if count % 100 == 0:
                print("Sim-time: ", sim_time)
                print("Number of contacts: ", torch.count_nonzero(contact_sensor.data.current_air_time == 0.0).item())
                print("-" * 80)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
