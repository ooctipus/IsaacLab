# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates different single-arm manipulators.

.. code-block:: bash

    # Usage with default PhysX physics and no visualizer.
    uv run python scripts/demos/arms.py

    # Usage with Newton visualizer and default PhysX physics.
    uv run python scripts/demos/arms.py visualizer=newton_gl

    # Usage with Newton (MJWarp) physics and no visualizer.
    uv run python scripts/demos/arms.py physics=newton_mjwarp

    # Usage with Newton visualizer and Newton (MJWarp) physics.
    uv run python scripts/demos/arms.py visualizer=newton_gl physics=newton_mjwarp

"""

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendCameraCfg, MultiBackendSimulationCfg

parser = argparse.ArgumentParser(
    description="This script demonstrates different single-arm manipulators.",
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics import MJWarpSolverCfg  # isort:skip
from isaaclab_physx.physics import PhysxCfg  # isort:skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip
from isaaclab_assets.robots.kinova import KINOVA_GEN3_N7_CFG, KINOVA_JACO2_N6S300_CFG, KINOVA_JACO2_N7S300_CFG  # isort:skip
from isaaclab_assets.robots.sawyer import SAWYER_CFG  # isort:skip
from isaaclab_assets.robots.universal_robots import UR10_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.assets import Articulation


@configclass
class DemoCfg:
    """Manipulator demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
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
    tables: list[AssetBaseCfg] = [
        AssetBaseCfg(
            prim_path="/World/Origin1/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.45, -2.0, 1.05)),
        ),
        AssetBaseCfg(
            prim_path="/World/Origin2/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, -2.0, 1.03)),
        ),
        AssetBaseCfg(
            prim_path="/World/Origin3/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, 0.0, 0.8)),
        ),
        AssetBaseCfg(
            prim_path="/World/Origin4/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 0.8)),
        ),
        AssetBaseCfg(
            prim_path="/World/Origin5/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.45, 2.0, 1.05)),
        ),
        AssetBaseCfg(
            prim_path="/World/Origin6/Table",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 2.0, 1.03)),
        ),
    ]
    franka: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Origin1/Robot")
    franka.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    franka.init_state.pos = (-1.0, -2.0, 1.05)
    ur10: ArticulationCfg = UR10_CFG.replace(prim_path="/World/Origin2/Robot")
    ur10.init_state.pos = (1.0, -2.0, 1.03)
    kinova_j2n7s300: ArticulationCfg = KINOVA_JACO2_N7S300_CFG.replace(prim_path="/World/Origin3/Robot")
    kinova_j2n7s300.init_state.pos = (-1.0, 0.0, 0.8)
    kinova_j2n6s300: ArticulationCfg = KINOVA_JACO2_N6S300_CFG.replace(prim_path="/World/Origin4/Robot")
    kinova_j2n6s300.init_state.pos = (1.0, 0.0, 0.8)
    kinova_gen3n7: ArticulationCfg = KINOVA_GEN3_N7_CFG.replace(prim_path="/World/Origin5/Robot")
    kinova_gen3n7.init_state.pos = (-1.0, 2.0, 1.05)
    sawyer: ArticulationCfg = SAWYER_CFG.replace(prim_path="/World/Origin6/Robot")
    sawyer.init_state.pos = (1.0, 2.0, 1.03)


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
        # update sim-time
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    cfg = resolve_config(DemoCfg(), config_overrides)
    with launch_simulation(cfg, args_cli):
        # Initialize the simulation context
        sim = sim_utils.SimulationContext(cfg.sim)
        # Set main camera
        sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
        asset_cfgs = tuple(
            asset_cfg
            for asset_cfg in (
                cfg.ground,
                cfg.light,
                *cfg.tables,
                cfg.franka,
                cfg.ur10,
                cfg.kinova_j2n7s300,
                cfg.kinova_j2n6s300,
                cfg.kinova_gen3n7,
                cfg.sawyer,
                cfg.camera,
            )
            if asset_cfg is not None
        )
        with ReplicateSession(asset_cfgs, 1, 0.0):
            _camera = cfg.camera.class_type(cfg.camera) if cfg.camera is not None else None
            for static_cfg in (cfg.ground, cfg.light, *cfg.tables):
                static_cfg.class_type(static_cfg)
            scene_entities = {
                name: robot_cfg.class_type(robot_cfg)
                for name, robot_cfg in (
                    ("franka_panda", cfg.franka),
                    ("ur10", cfg.ur10),
                    ("kinova_j2n7s300", cfg.kinova_j2n7s300),
                    ("kinova_j2n6s300", cfg.kinova_j2n6s300),
                    ("kinova_gen3n7", cfg.kinova_gen3n7),
                    ("sawyer", cfg.sawyer),
                )
            }
        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")
        # Run the simulator
        run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
