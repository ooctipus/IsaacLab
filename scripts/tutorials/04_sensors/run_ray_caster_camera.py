# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows how to use the ray-cast camera sensor from the Isaac Lab framework.

The camera sensor is based on using Warp kernels which do ray-casting against static meshes.

.. code-block:: bash

    uv run python scripts/tutorials/04_sensors/run_ray_caster_camera.py visualizer=kit

"""

import argparse

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendSimulationCfg

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the ray-cast camera sensor.")
parser.add_argument("--save", action="store_true", default=False, help="Save the obtained data to disk.")
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import os
from typing import Any

import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import ReplicateSession
from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import project_points, unproject_depth


@configclass
class TutorialCfg:
    """Ray-cast camera tutorial configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(device=args_cli.device, physics=PhysxCfg())
    num_envs: int = 2
    env_spacing: float = 1.0
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd"),
    )
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75)),
    )
    camera_frame: AssetBaseCfg = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/CameraSensor", spawn=sim_utils.SensorFrameCfg())
    camera: RayCasterCameraCfg = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraSensor",
        mesh_prim_paths=["/World/ground"],
        update_period=0.1,
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        data_types=["distance_to_image_plane", "normals", "distance_to_camera"],
        debug_vis=True,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=480,
            width=640,
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Run the simulator."""
    # extract entities for simplified notation
    camera: RayCasterCamera = scene_entities["camera"]

    # Create the Replicator writer only when saving. The ray-cast camera itself
    # is Warp-based and does not require Replicator or RTX rendering extensions.
    rep_writer: Any | None = None
    if args_cli.save:
        import omni.replicator.core as rep

        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ray_caster_camera")
        rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    # Set pose: There are two ways to set the pose of the camera.
    # -- Option-1: Set pose using view
    eyes = torch.tensor([[2.5, 2.5, 2.5], [-2.5, -2.5, 2.5]], device=sim.device)
    targets = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device=sim.device)
    camera.set_world_poses_from_view(eyes, targets)
    # -- Option-2: Set pose using ROS
    # position = torch.tensor([[2.5, 2.5, 2.5]], device=sim.device)
    # orientation = torch.tensor([[-0.17591989, 0.33985114, 0.82047325, -0.42470819]], device=sim.device)
    # camera.set_world_poses(position, orientation, indices=[0], convention="ros")

    # Simulate physics
    while sim.is_headless_or_exist_active_visualizer():
        # Step simulation
        sim.step()
        # Update camera data
        camera.update(dt=sim.get_physics_dt())

        # Print camera info
        print(camera)
        print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        print("-------------------------------")

        # Extract camera data
        if args_cli.save:
            # Extract camera data
            camera_index = 0
            # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
            single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )
            # Pack data back into replicator format to save them using its writer
            rep_output = {"annotators": {}}
            for key, data in single_cam_data.items():
                info = camera.data.info.get(key)
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            # Save images
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            assert rep_writer is not None
            rep_writer.write(rep_output)

            # Pointcloud in world frame
            points_3d_cam = unproject_depth(
                camera.data.output["distance_to_image_plane"], camera.data.intrinsic_matrices
            )

            # Check methods are valid
            im_height, im_width = camera.image_shape
            # -- project points to (u, v, d)
            reproj_points = project_points(points_3d_cam, camera.data.intrinsic_matrices)
            reproj_depths = reproj_points[..., -1].view(-1, im_width, im_height).transpose_(1, 2)
            sim_depths = camera.data.output["distance_to_image_plane"].squeeze(-1)
            torch.testing.assert_close(reproj_depths, sim_depths)


def main():
    """Main function."""
    cfg = resolve_config(TutorialCfg(), config_overrides)
    with launch_simulation(cfg, args_cli):
        sim = sim_utils.SimulationContext(cfg.sim)
        sim.set_camera_view([2.5, 2.5, 3.5], [0.0, 0.0, 0.0])
        with ReplicateSession(
            (cfg.ground, cfg.light, cfg.camera_frame, cfg.camera, cfg.camera.visualizer_cfg),
            cfg.num_envs,
            cfg.env_spacing,
        ):
            cfg.ground.class_type(cfg.ground)
            cfg.light.class_type(cfg.light)
            cfg.camera_frame.class_type(cfg.camera_frame)
            scene_entities = {"camera": cfg.camera.class_type(cfg.camera)}
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim=sim, scene_entities=scene_entities)


if __name__ == "__main__":
    main()
