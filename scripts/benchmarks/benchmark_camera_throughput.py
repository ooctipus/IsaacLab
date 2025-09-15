# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark throughput of different camera implementations.

This script benchmarks per-step time while varying:
- camera implementation: standard camera, tiled camera, warp ray-caster camera
- image resolutions (height x width)
- number of environments

Sensors are added to the scene config before `InteractiveScene` is constructed.
Each benchmark run initializes a fresh simulation and scene and tears it down.

Examples:

  - Benchmark all camera types across resolutions:
      ./isaaclab.sh -p scripts/benchmarks/benchmark_camera_throughput.py \\
        --num_envs 256 512 --impls standard,tiled,ray_caster \\
        --resolutions 240x320,480x640 --steps 200 --warmup 20 --headless

  - Only standard camera at 720p:
      ./isaaclab.sh -p scripts/benchmarks/benchmark_camera_throughput.py \\
        --num_envs 256 --impls standard --resolutions 720x1280 --steps 200 --warmup 20 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark throughput of different camera implementations.")
parser.add_argument(
    "--num_envs",
    type=int,
    nargs="+",
    default=[256, 512, 1024],
    help="List of environment counts to benchmark (e.g., 256 512 1024).",
)
parser.add_argument(
    "--impls",
    type=str,
    default="standard,tiled,ray_caster",
    help="Comma-separated list of implementations: standard,tiled,ray_caster",
)
parser.add_argument(
    "--resolutions",
    type=str,
    default="240x320,480x640",
    help="Comma-separated list of HxW resolutions, e.g., 240x320,480x640",
)
parser.add_argument("--steps", type=int, default=500, help="Steps per run to time.")
parser.add_argument("--warmup", type=int, default=50, help="Warmup steps per run before timing.")

# Append AppLauncher CLI args and parse
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Robot config to attach sensors under a valid prim
from isaaclab_assets.robots.anymal import ANYMAL_D_CFG  # isort: skip


def _parse_resolutions(res_str: str) -> list[tuple[int, int]]:
    resolutions: list[tuple[int, int]] = []
    for token in [s for s in res_str.split(",") if s]:
        h, w = token.lower().split("x")
        resolutions.append((int(h), int(w)))
    return resolutions


@configclass
class CameraBenchmarkSceneCfg(InteractiveSceneCfg):
    """Scene config with ground, light, robot, and one camera sensor per env."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd"),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore[attr-defined]

    # one cube per environment (optional target for ray-caster camera)
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    standard_camera: CameraCfg | None = None
    tiled_camera: TiledCameraCfg | None = None
    ray_caster_camera: RayCasterCameraCfg | None = None


def _make_scene_cfg_standard(num_envs: int, height: int, width: int, debug_vis: bool) -> CameraBenchmarkSceneCfg:
    scene_cfg = CameraBenchmarkSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.standard_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        debug_vis=debug_vis,
    )
    return scene_cfg


def _make_scene_cfg_tiled(num_envs: int, height: int, width: int, debug_vis: bool) -> CameraBenchmarkSceneCfg:
    scene_cfg = CameraBenchmarkSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/TiledCamera",
        height=height,
        width=width,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1e4)
        ),
        debug_vis=debug_vis,
    )
    return scene_cfg


def _make_scene_cfg_ray_caster(num_envs: int, height: int, width: int, debug_vis: bool) -> CameraBenchmarkSceneCfg:
    scene_cfg = CameraBenchmarkSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene_cfg.ray_caster_camera = RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",  # attach to existing prim
        mesh_prim_paths=["/World/ground", "/World/envs/env_.*/cube"],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0, horizontal_aperture=20.955, height=height, width=width
        ),
        data_types=["distance_to_image_plane"],
        debug_vis=debug_vis,
    )
    return scene_cfg


import isaacsim.core.utils.stage as stage_utils


def _setup_scene(scene_cfg: CameraBenchmarkSceneCfg) -> tuple[SimulationContext, InteractiveScene, float]:
    # Create a new stage to avoid residue across runs
    stage_utils.create_new_stage()
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))
    setup_time_begin = time.perf_counter_ns()
    scene = InteractiveScene(scene_cfg)
    setup_time_end = time.perf_counter_ns()
    print(f"[INFO]: Scene creation time: {(setup_time_end - setup_time_begin) / 1e6:.2f} ms")
    reset_time_begin = time.perf_counter_ns()
    sim.reset()
    reset_time_end = time.perf_counter_ns()
    print(f"[INFO]: Sim start time: {(reset_time_end - reset_time_begin) / 1e6:.2f} ms")
    return sim, scene, sim.get_physics_dt()


def main():
    impls = [s.strip() for s in args_cli.impls.split(",") if s]
    resolutions = _parse_resolutions(args_cli.resolutions)
    results: list[dict[str, object]] = []

    def _bench(num_envs: int, impl: str, height: int, width: int):
        if impl == "standard":
            scene_cfg = _make_scene_cfg_standard(num_envs, height, width, debug_vis=not args_cli.headless)
            sim, scene, sim_dt = _setup_scene(scene_cfg)
            camera_obj = scene["standard_camera"]
            label = "StandardCamera"
        elif impl == "tiled":
            scene_cfg = _make_scene_cfg_tiled(num_envs, height, width, debug_vis=not args_cli.headless)
            sim, scene, sim_dt = _setup_scene(scene_cfg)
            camera_obj = scene["tiled_camera"]
            label = "TiledCamera"
        elif impl == "ray_caster":
            scene_cfg = _make_scene_cfg_ray_caster(num_envs, height, width, debug_vis=not args_cli.headless)
            sim, scene, sim_dt = _setup_scene(scene_cfg)
            camera_obj = scene["ray_caster_camera"]
            label = "RayCasterCamera"
        else:
            raise ValueError(f"Unknown impl: {impl}")

        # Warmup
        for _ in range(args_cli.warmup):
            sim.step()
            camera_obj.update(dt=sim_dt)
        # Timing
        t0 = time.perf_counter_ns()
        for _ in range(args_cli.steps):
            sim.step()
            camera_obj.update(dt=sim_dt)
        t1 = time.perf_counter_ns()
        per_step_ms = (t1 - t0) / args_cli.steps / 1e6
        print(f"[INFO]: {label}: {num_envs} envs, res={height}x{width}, per-step={per_step_ms:.3f} ms")
        results.append({
            "impl": impl,
            "num_envs": num_envs,
            "height": height,
            "width": width,
            "per_step_ms": float(per_step_ms),
        })
        # Teardown
        sim.clear_instance()

    for num_envs in args_cli.num_envs:
        for impl in impls:
            print(f"\n[INFO]: Benchmarking {impl} cameras with {num_envs} envs")
            for h, w in resolutions:
                _bench(num_envs, impl, h, w)

    # Save results
    os.makedirs("outputs/benchmarks", exist_ok=True)
    csv_path = os.path.join("outputs/benchmarks", "camera_throughput.csv")
    md_path = os.path.join("outputs/benchmarks", "camera_throughput.md")

    fieldnames = ["impl", "num_envs", "height", "width", "per_step_ms"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    with open(md_path, "w") as f:
        f.write("| impl | num_envs | height | width | per_step_ms |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in results:
            f.write(f"| {r['impl']} | {r['num_envs']} | {r['height']} | {r['width']} | {r['per_step_ms']:.3f} |\n")
    print(f"[INFO]: Saved benchmark results to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
