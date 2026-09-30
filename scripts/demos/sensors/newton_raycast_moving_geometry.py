# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: Newton BVH ray-cast sensor over moving geometry (live BVH refit).

A slowly spinning grid ray-cast sensor hovers above a scene where boxes keep
falling and a long bar sweeps around kinematically. Every hit tracks the moving
bodies because the scene BVH is refit each step inside the shared CUDA graph.
The sensor publishes its debug visualization through the configured visualizer.

.. code-block:: bash

    uv run python scripts/demos/sensors/newton_raycast_moving_geometry.py

"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendSimulationCfg

parser = argparse.ArgumentParser(
    description="Newton BVH ray-cast sensor over moving geometry.",
    conflict_handler="resolve",
)
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import math

import torch
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.sensors import NewtonRaycastSensorCfg

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

BOX_DROP_POSITIONS = [(1.0, 0.6, 3.0), (-0.8, -1.0, 3.5), (0.2, -1.2, 4.0)]


def _falling_box_cfg(index: int) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Box_{index}",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4 + 0.2 * index, 0.9 - 0.3 * index)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_DROP_POSITIONS[index]),
    )


@configclass
class MovingGeometrySceneCfg(InteractiveSceneCfg):
    """Ground plane, falling boxes, a sweeping bar, and a hovering sensor."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    box_0 = _falling_box_cfg(0)
    box_1 = _falling_box_cfg(1)
    box_2 = _falling_box_cfg(2)

    bar = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Bar",
        spawn=sim_utils.CuboidCfg(
            size=(3.5, 0.3, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),
    )

    body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        spawn=sim_utils.CuboidCfg(
            size=(0.3, 0.3, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.6, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.5)),
    )

    raycast = NewtonRaycastSensorCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        # Start the rays below the carrier body so they do not hit it.
        offset=NewtonRaycastSensorCfg.OffsetCfg(pos=(0.0, 0.0, -0.1)),
        pattern_cfg=GridPatternCfg(resolution=0.25, size=(3.0, 3.0)),
        ray_alignment="yaw",
        max_distance=10.0,
        debug_vis=True,
    )


@configclass
class DemoCfg:
    """Moving-geometry ray-cast demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=1 / 100,
        device=args_cli.device,
        physics=MJWarpSolverCfg(),
    )
    scene: MovingGeometrySceneCfg = MovingGeometrySceneCfg(num_envs=1, env_spacing=1.0)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Spin the sensor, sweep the bar, and keep dropping boxes through the rays."""
    boxes: list[RigidObject] = [scene[f"box_{i}"] for i in range(len(BOX_DROP_POSITIONS))]
    bar: RigidObject = scene["bar"]
    body: RigidObject = scene["body"]
    sim_dt = sim.get_physics_dt()
    zero_vel = torch.zeros(1, 6, device=sim.device)
    zero_angle = torch.zeros(1, device=sim.device)
    count = 0
    while sim.is_headless_or_exist_active_visualizer():
        # Re-drop the boxes periodically so geometry keeps moving through the rays.
        if count % 400 == 0:
            for box, drop_pos in zip(boxes, BOX_DROP_POSITIONS):
                pose = torch.tensor([[*drop_pos, 0.3, 0.3, 0.0, 0.9]], device=sim.device)
                pose[:, 3:] /= torch.linalg.norm(pose[:, 3:])
                box.write_root_pose_to_sim_index(root_pose=pose)
                box.write_root_velocity_to_sim_index(root_velocity=zero_vel)
                box.reset()

        t = count * sim_dt
        # Kinematically sweep the bar and spin the sensor body about z.
        bar_quat = math_utils.quat_from_euler_xyz(zero_angle, zero_angle, zero_angle + 0.8 * t)
        bar_pos = torch.tensor([[0.0, 0.0, 0.8]], device=sim.device)
        bar.write_root_pose_to_sim_index(root_pose=torch.cat([bar_pos, bar_quat], dim=-1))
        bar.write_root_velocity_to_sim_index(root_velocity=zero_vel)

        body_quat = math_utils.quat_from_euler_xyz(zero_angle, zero_angle, zero_angle - 0.3 * t)
        body_pos = torch.tensor(
            [[0.6 * math.cos(0.5 * t), 0.6 * math.sin(0.5 * t), 2.5]],
            device=sim.device,
        )
        body.write_root_pose_to_sim_index(root_pose=torch.cat([body_pos, body_quat], dim=-1))
        body.write_root_velocity_to_sim_index(root_velocity=zero_vel)
        scene.write_data_to_sim()

        sim.step()
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    cfg = resolve_config(DemoCfg(), config_overrides)
    with launch_simulation(cfg.sim, args_cli):
        sim = sim_utils.SimulationContext(cfg.sim)
        sim.set_camera_view(eye=[6.0, 6.0, 4.5], target=[0.0, 0.0, 1.0])
        scene = cfg.scene.class_type(cfg.scene)
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
