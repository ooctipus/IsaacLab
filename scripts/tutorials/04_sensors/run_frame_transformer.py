# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the FrameTransformer sensor by visualizing the frames that it creates.

.. code-block:: bash

    # Usage
    uv run python scripts/tutorials/04_sensors/run_frame_transformer.py visualizer=kit

"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendSimulationCfg

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script checks the FrameTransformer sensor by visualizing the frames that it creates."
)
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.cloner import ReplicateSession
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import FrameTransformer

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip

ROBOT_PRIM_PATH_EXPR = "/World/envs/env_[^/]+/Robot"
_ROT_OFFSET = math_utils.quat_from_euler_xyz(torch.zeros(1), torch.zeros(1), torch.tensor(-math.pi / 2))
_POS_OFFSET = math_utils.quat_apply(_ROT_OFFSET, torch.tensor([0.08795, 0.01305, -0.33797]))


@configclass
class TutorialCfg:
    """Frame-transformer tutorial configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(dt=0.005, device=args_cli.device, physics=PhysxCfg())
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_transformer: FrameTransformerCfg = FrameTransformerCfg(
        prim_path=f"{ROBOT_PRIM_PATH_EXPR}/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path=f"{ROBOT_PRIM_PATH_EXPR}/[^/]+"),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{ROBOT_PRIM_PATH_EXPR}/LF_SHANK",
                name="LF_FOOT_USER",
                offset=OffsetCfg(pos=tuple(_POS_OFFSET.tolist()), rot=tuple(_ROT_OFFSET[0].tolist())),
            ),
        ],
        debug_vis=False,
    )
    frame_visualizer: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/FrameVisualizerFromScript")
    frame_visualizer.markers["frame"].scale = (0.1, 0.1, 0.1)


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # extract entities for simplified notation
    robot: Articulation = scene_entities["robot"]
    frame_transformer: FrameTransformer = scene_entities["frame_transformer"]

    # We only want one visualization at a time. This visualizer will be used
    # to step through each frame so the user can verify that the correct frame
    # is being visualized as the frame names are printing to console
    transform_visualizer = scene_entities["frame_visualizer"]

    frame_index = 0
    # Simulate physics
    while sim.is_headless_or_exist_active_visualizer():
        # perform this loop at policy control freq (50 Hz)
        robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch.clone())
        robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # read data from sim
        robot.update(sim_dt)
        frame_transformer.update(dt=sim_dt)

        # Change the frame that we are visualizing to ensure that frame names
        # are correctly associated with the frames
        if count % 50 == 0:
            frame_names = frame_transformer.data.target_frame_names
            frame_index = (frame_index + 1) % len(frame_names)
            print(f"Displaying Frame ID {frame_index}: {frame_names[frame_index]}")

        source_pos = frame_transformer.data.source_pos_w.torch
        source_quat = frame_transformer.data.source_quat_w.torch
        target_pos = frame_transformer.data.target_pos_w.torch[:, frame_index]
        target_quat = frame_transformer.data.target_quat_w.torch[:, frame_index]
        transform_visualizer.visualize(
            torch.cat([source_pos, target_pos], dim=0), torch.cat([source_quat, target_quat], dim=0)
        )


def main():
    """Main function."""
    cfg = resolve_config(TutorialCfg(), config_overrides)
    with launch_simulation(cfg, args_cli):
        sim = sim_utils.SimulationContext(cfg.sim)
        sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
        with ReplicateSession(
            (
                cfg.ground,
                cfg.light,
                cfg.robot,
                cfg.frame_transformer,
                cfg.frame_transformer.visualizer_cfg,
                cfg.frame_visualizer,
            ),
            1,
            0.0,
        ):
            cfg.ground.class_type(cfg.ground)
            cfg.light.class_type(cfg.light)
            scene_entities = {
                "robot": cfg.robot.class_type(cfg.robot),
                "frame_transformer": cfg.frame_transformer.class_type(cfg.frame_transformer),
                "frame_visualizer": cfg.frame_visualizer.class_type(cfg.frame_visualizer),
            }
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene_entities)


if __name__ == "__main__":
    main()
