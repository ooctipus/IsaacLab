# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn a pile of cables that collide and settle on each other.

.. code-block:: bash

    # Usage with default Newton VBD physics and no visualizer.
    uv run --extra isaacsim python scripts/demos/cables.py

    # Usage with explicit Newton VBD physics and Newton visualizer.
    uv run python scripts/demos/cables.py physics=newton_vbd visualizer=newton_gl

"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendCameraCfg, MultiBackendSimulationCfg

parser = argparse.ArgumentParser(description="Spawn a pile of cables with Newton VBD.", conflict_handler="resolve")
parser.add_argument("--num_cables", type=int, default=25, help="Number of cables to spawn.")
parser.add_argument("--num_segments", type=int, default=20, help="Number of segments per cable.")
parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many steps; negative runs forever.")
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

if args_cli.num_cables < 1:
    parser.error("--num_cables must be at least 1.")
if args_cli.num_segments < 2:
    parser.error("--num_segments must be at least 2.")

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, CableObjectCfg
from isaaclab.cloner import CloneCfg, ReplicateSession
from isaaclab.utils.configclass import configclass

from isaaclab_newton.physics import VBDSolverCfg  # isort: skip

if TYPE_CHECKING:
    from isaaclab.assets import CableObject


def _cable_cfgs(num_cables: int, num_segments: int) -> dict[str, CableObjectCfg]:
    """Return the cable pile as configuration data."""
    cable_length = 0.5
    segment_length = cable_length / num_segments
    thickness = 0.01
    radius = 0.5 * thickness
    stretch_modulus = 5.0e5 * segment_length / (math.pi * radius**2)
    bend_modulus = 20.0 * segment_length / (0.25 * math.pi * radius**4)
    positions = [(index * segment_length, 0.0, 0.0) for index in range(num_segments + 1)]
    configs = {}
    for index in range(num_cables):
        angle = random.uniform(0.0, 2.0 * math.pi)
        configs[f"cable_{index:03d}"] = CableObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Cable{index:03d}",
            spawn=sim_utils.CableCfg(
                positions=positions,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(random.random(), random.random(), random.random())
                ),
                physics_material=sim_utils.CableMaterialCfg(
                    thickness=thickness,
                    density=100.0,
                    stretch_stiffness=stretch_modulus,
                    bend_stiffness=bend_modulus,
                ),
                collision_props=[sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True)],
            ),
            init_state=CableObjectCfg.InitialStateCfg(
                pos=(
                    random.uniform(-0.3, 0.3) - 0.5 * cable_length * math.cos(angle),
                    random.uniform(-0.3, 0.3) - 0.5 * cable_length * math.sin(angle),
                    0.8 + index * 1.5 * thickness,
                ),
                rot=(0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)),
            ),
        )
    return configs


@configclass
class DemoCfg:
    """Cable-pile demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=0.01,
        device=args_cli.device,
        physics=preset(
            default=VBDSolverCfg(iterations=20, num_substeps=8),
            newton_vbd=VBDSolverCfg(iterations=20, num_substeps=8),
        ),
    )
    camera: MultiBackendCameraCfg = MultiBackendCameraCfg()
    num_envs: int = 1
    env_spacing: float = 2.0
    clone_cfg: CloneCfg = CloneCfg()
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    cables: dict[str, CableObjectCfg] = MISSING


def reset_cables(entities: dict[str, CableObject]) -> None:
    """Restore every cable to its initial segment state."""
    for cable in entities.values():
        cable.write_segment_pose_to_sim_index(segment_pose=cable.data.default_segment_pose_w)
        cable.write_segment_velocity_to_sim_index(segment_velocity=cable.data.default_segment_velocity_w)


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, CableObject], max_steps: int = -1) -> None:
    """Run the simulation and periodically restore the cable pile."""
    sim_dt = sim.get_physics_dt()
    reset_steps = max(1, int(2.0 / sim_dt))
    count = 0

    while (max_steps < 0 or count < max_steps) and sim.is_headless_or_exist_active_visualizer():
        if count > 0 and count % reset_steps == 0:
            reset_cables(entities)
            print("[INFO]: Resetting cable state...")

        sim.step(render=False)
        for cable in entities.values():
            cable.update(sim_dt)
        if sim.is_rendering:
            sim.render()
        count += 1


def main() -> None:
    """Launch and run the cable pile demo."""
    cfg = resolve_config(DemoCfg(cables=_cable_cfgs(args_cli.num_cables, args_cli.num_segments)), config_overrides)
    with launch_simulation(cfg, args_cli):
        sim = sim_utils.SimulationContext(cfg.sim)
        sim.set_camera_view(eye=(2.0, 2.0, 1.0), target=(0.0, 0.0, 0.25))
        asset_cfgs = tuple(
            asset_cfg
            for asset_cfg in (cfg.ground, cfg.light, *cfg.cables.values(), cfg.camera)
            if asset_cfg is not None
        )
        with ReplicateSession(
            asset_cfgs,
            num_clones=cfg.num_envs,
            env_spacing=cfg.env_spacing,
            clone_strategy=cfg.clone_cfg.clone_strategy,
            env_template=cfg.clone_cfg.clone_template,
            replicate_physics=cfg.clone_cfg.replicate_physics,
        ):
            _camera = cfg.camera.class_type(cfg.camera) if cfg.camera is not None else None
            cfg.ground.class_type(cfg.ground)
            cfg.light.class_type(cfg.light)
            entities = {name: cable_cfg.class_type(cable_cfg) for name, cable_cfg in cfg.cables.items()}
        sim.reset()
        print("[INFO]: Setup complete...")
        run_simulator(sim, entities, args_cli.max_steps)


if __name__ == "__main__":
    main()
