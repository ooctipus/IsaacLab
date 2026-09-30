# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo: Newton BVH ray-cast sensor scanning a heightfield terrain.

A grid ray-cast sensor rides a body that circles, bobs, and tumbles above a
wave heightfield. The sensor publishes its debug visualization through the
configured visualizer.

.. code-block:: bash

    uv run python scripts/demos/sensors/newton_raycast_heightfield.py

"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendSimulationCfg

parser = argparse.ArgumentParser(
    description="Newton BVH ray-cast sensor scanning a heightfield.",
    conflict_handler="resolve",
)
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import math

import torch
from isaaclab_newton.physics import MJWarpSolverCfg
from isaaclab_newton.sensors import NewtonRaycastSensorCfg

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass

WAVE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=1.0,
    num_rows=1,
    num_cols=1,
    use_cache=False,
    sub_terrains={
        "waves": terrain_gen.HfWaveTerrainCfg(amplitude_range=(0.25, 0.25), num_waves=6),
    },
)


@configclass
class HeightfieldSceneCfg(InteractiveSceneCfg):
    """Wave heightfield with a floating sensor body."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator", terrain_generator=WAVE_TERRAIN_CFG
    )

    body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.25, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.6, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)),
    )

    raycast = NewtonRaycastSensorCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        pattern_cfg=GridPatternCfg(resolution=0.25, size=(1.5, 1.0)),
        ray_alignment="base",
        global_world_only=True,
        max_distance=10.0,
        debug_vis=True,
    )


@configclass
class DemoCfg:
    """Heightfield ray-cast demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=1 / 100,
        device=args_cli.device,
        physics=MJWarpSolverCfg(),
    )
    scene: HeightfieldSceneCfg = HeightfieldSceneCfg(num_envs=1, env_spacing=1.0)


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Fly the sensor body over the terrain and visualize the rays."""
    body: RigidObject = scene["body"]
    sim_dt = sim.get_physics_dt()
    zero_vel = torch.zeros(1, 6, device=sim.device)
    t = 0.0
    while sim.is_headless_or_exist_active_visualizer():
        # Circle above the terrain while bobbing, pitching, rolling, and yawing.
        angle = 0.4 * t
        pos = torch.tensor(
            [[3.0 * math.cos(angle), 3.0 * math.sin(angle), 1.4 + 0.3 * math.sin(0.9 * t)]], device=sim.device
        )
        angles = torch.tensor([0.3 * math.sin(0.7 * t), 0.25 * math.sin(1.1 * t), angle + math.pi / 2.0])
        quat = math_utils.quat_from_euler_xyz(*(a.unsqueeze(0) for a in angles.to(sim.device)))
        body.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1))
        body.write_root_velocity_to_sim_index(root_velocity=zero_vel)
        scene.write_data_to_sim()

        sim.step()
        scene.update(sim_dt)
        t += sim_dt


def main():
    """Main function."""
    cfg = resolve_config(DemoCfg(), config_overrides)
    with launch_simulation(cfg.sim, args_cli):
        sim = sim_utils.SimulationContext(cfg.sim)
        sim.set_camera_view(eye=[7.0, 7.0, 5.0], target=[0.0, 0.0, 0.0])
        scene = cfg.scene.class_type(cfg.scene)
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
