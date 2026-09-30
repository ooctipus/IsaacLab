# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates different legged robots.

.. code-block:: bash

    # Usage with default PhysX physics and no visualizer.
    uv run python scripts/demos/quadrupeds.py

    # Usage with Newton visualizer and default PhysX physics.
    uv run python scripts/demos/quadrupeds.py visualizer=newton_gl

    # Usage with Newton (MJWarp) physics and no visualizer.
    uv run python scripts/demos/quadrupeds.py physics=newton_mjwarp

    # Usage with Newton visualizer and Newton (MJWarp) physics.
    uv run python scripts/demos/quadrupeds.py visualizer=newton_gl physics=newton_mjwarp

"""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendCameraCfg, MultiBackendSimulationCfg

parser = argparse.ArgumentParser(
    description="This script demonstrates different legged robots.",
    conflict_handler="resolve",
)
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.cloner import ReplicateSession

##
# Pre-defined configs
##
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics import MJWarpSolverCfg  # isort:skip
from isaaclab_physx.physics import PhysxCfg  # isort:skip
from isaaclab_assets.robots.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG, ANYMAL_D_CFG  # isort:skip
from isaaclab_assets.robots.spot import SPOT_CFG  # isort:skip
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG, UNITREE_GO1_CFG, UNITREE_GO2_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class DemoCfg:
    """Quadruped demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=1 / 200,
        device=args_cli.device,
        physics=preset(default=PhysxCfg(), isaacsim_physx=PhysxCfg(), newton_mjwarp=MJWarpSolverCfg()),
    )
    camera: MultiBackendCameraCfg = MultiBackendCameraCfg()
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    anymal_b: ArticulationCfg = ANYMAL_B_CFG.replace(prim_path="/World/Origin1/Robot")
    anymal_b.init_state.pos = (-1.875, -0.625, anymal_b.init_state.pos[2])
    anymal_c: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/Origin2/Robot")
    anymal_c.init_state.pos = (-0.625, -0.625, anymal_c.init_state.pos[2])
    anymal_d: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="/World/Origin3/Robot")
    anymal_d.init_state.pos = (0.625, -0.625, anymal_d.init_state.pos[2])
    unitree_a1: ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="/World/Origin4/Robot")
    unitree_a1.init_state.pos = (1.875, -0.625, unitree_a1.init_state.pos[2])
    unitree_go1: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="/World/Origin5/Robot")
    unitree_go1.init_state.pos = (-1.875, 0.625, unitree_go1.init_state.pos[2])
    unitree_go2: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/Origin6/Robot")
    unitree_go2.init_state.pos = (-0.625, 0.625, unitree_go2.init_state.pos[2])
    spot: ArticulationCfg = SPOT_CFG.replace(prim_path="/World/Origin7/Robot")
    spot.init_state.pos = (0.625, 0.625, spot.init_state.pos[2])


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
    cfg = resolve_config(DemoCfg(), config_overrides)
    with launch_simulation(cfg, args_cli):
        sim = sim_utils.SimulationContext(cfg.sim)
        sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
        asset_cfgs = tuple(
            asset_cfg
            for asset_cfg in (
                cfg.ground,
                cfg.light,
                cfg.anymal_b,
                cfg.anymal_c,
                cfg.anymal_d,
                cfg.unitree_a1,
                cfg.unitree_go1,
                cfg.unitree_go2,
                cfg.spot,
                cfg.camera,
            )
            if asset_cfg is not None
        )
        with ReplicateSession(asset_cfgs, 1, 0.0):
            _camera = cfg.camera.class_type(cfg.camera) if cfg.camera is not None else None
            cfg.ground.class_type(cfg.ground)
            cfg.light.class_type(cfg.light)
            scene_entities = {
                name: robot_cfg.class_type(robot_cfg)
                for name, robot_cfg in (
                    ("anymal_b", cfg.anymal_b),
                    ("anymal_c", cfg.anymal_c),
                    ("anymal_d", cfg.anymal_d),
                    ("unitree_a1", cfg.unitree_a1),
                    ("unitree_go1", cfg.unitree_go1),
                    ("unitree_go2", cfg.unitree_go2),
                    ("spot", cfg.spot),
                )
            }
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene_entities)


if __name__ == "__main__":
    main()
