# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates different dexterous hands.

.. code-block:: bash

    # Usage with default PhysX physics and no visualizer.
    uv run python scripts/demos/hands.py

    # Usage with Newton visualizer and default PhysX physics.
    uv run python scripts/demos/hands.py visualizer=newton_gl

    # Usage with Newton (MJWarp) physics and no visualizer.
    uv run python scripts/demos/hands.py physics=newton_mjwarp

    # Usage with Newton visualizer and Newton (MJWarp) physics.
    uv run python scripts/demos/hands.py visualizer=newton_gl physics=newton_mjwarp

"""

import argparse
from typing import TYPE_CHECKING

import torch

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendCameraCfg, MultiBackendSimulationCfg

parser = argparse.ArgumentParser(
    description="This script demonstrates different dexterous hands.",
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
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG  # isort:skip
from isaaclab_assets.robots.shadow_hand import (
    SHADOW_HAND_NEWTON_CFG,
    SHADOW_HAND_PHYSX_CFG,
    TENDON_POSITION_LIMITS,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class DemoCfg:
    """Dexterous-hand demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=0.01,
        device=args_cli.device,
        physics=preset(
            default=PhysxCfg(),
            isaacsim_physx=PhysxCfg(),
            newton_mjwarp=MJWarpSolverCfg(
                njmax=200,
                nconmax=70,
                impratio=10.0,
                cone="elliptic",
                integrator="implicitfast",
                update_data_interval=2,
                ccd_iterations=50,
                num_substeps=2,
                debug_mode=False,
            ),
        ),
    )
    camera: MultiBackendCameraCfg = MultiBackendCameraCfg()
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    allegro: ArticulationCfg = ALLEGRO_HAND_CFG.replace(prim_path="/World/Origin1/Robot")
    allegro.init_state.pos = (-0.25, 0.0, allegro.init_state.pos[2])
    shadow_hand: ArticulationCfg = preset(
        default=SHADOW_HAND_PHYSX_CFG.replace(prim_path="/World/Origin2/Robot"),
        isaacsim_physx=SHADOW_HAND_PHYSX_CFG.replace(prim_path="/World/Origin2/Robot"),
        newton_mjwarp=SHADOW_HAND_NEWTON_CFG.replace(prim_path="/World/Origin2/Robot"),
    )


def run_simulator(sim: "sim_utils.SimulationContext", entities: dict[str, "Articulation"]):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Start with hand open
    grasp_mode = 0
    # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
    while sim.is_headless_or_exist_active_visualizer():
        # reset
        if count % 1000 == 0:
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
                joint_pos, joint_vel = (
                    robot.data.default_joint_pos.torch.clone(),
                    robot.data.default_joint_vel.torch.clone(),
                )
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                # reset the internal state
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # toggle grasp mode
        if count % 100 == 0:
            grasp_mode = 1 - grasp_mode
        # apply default actions to the hands robots
        for robot in entities.values():
            # generate joint positions
            joint_pos_target = robot.data.soft_joint_pos_limits.torch[..., grasp_mode]
            # apply action to the robot
            robot.set_joint_position_target_index(target=joint_pos_target)
            # A tendon has no actuator on its spanned joints, so it needs its own command.
            # Span comes from the asset: a fixed tendon authors no position limit of its own.
            if robot.num_fixed_tendons > 0:
                tendon_pos_target = torch.full(
                    (robot.num_instances, robot.num_fixed_tendons),
                    TENDON_POSITION_LIMITS[grasp_mode],
                    device=robot.device,
                )
                robot.set_fixed_tendon_position_target_index(target=tendon_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    cfg = resolve_config(DemoCfg(), config_overrides)
    cfg.shadow_hand.init_state = cfg.shadow_hand.init_state.replace(
        pos=(0.25, 0.0, 0.5),
        rot=(0.52296271, -0.47593067, 0.47593067, 0.52296271),
    )
    with launch_simulation(cfg, args_cli):
        # Initialize the simulation context
        sim = sim_utils.SimulationContext(cfg.sim)
        # Set main camera
        sim.set_camera_view(eye=[0.0, -0.5, 1.5], target=[0.0, -0.05, 0.45])
        asset_cfgs = tuple(
            asset_cfg
            for asset_cfg in (cfg.ground, cfg.light, cfg.allegro, cfg.shadow_hand, cfg.camera)
            if asset_cfg is not None
        )
        with ReplicateSession(asset_cfgs, 1, 0.0):
            _camera = cfg.camera.class_type(cfg.camera) if cfg.camera is not None else None
            cfg.ground.class_type(cfg.ground)
            cfg.light.class_type(cfg.light)
            scene_entities = {
                "allegro": cfg.allegro.class_type(cfg.allegro),
                "shadow_hand": cfg.shadow_hand.class_type(cfg.shadow_hand),
            }
        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        # Run the simulator
        run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main execution
    main()
