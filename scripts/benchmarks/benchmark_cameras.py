# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark renderer-backed or ray-caster cameras through the normal clone lifecycle.

The standalone workflow is deliberately a direct cfg example: assets and the camera are
declared in :class:`DirectBenchmarkCfg`, one :class:`ReplicateSession` owns every prototype and
clone, and runtime objects are constructed only through ``cfg.class_type(cfg)``.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default=None, help="Optional manager-based task receiving cameras.")
parser.add_argument("--camera_type", choices=("camera", "ray_caster"), default="camera")
parser.add_argument("--num_cameras", type=int, default=1)
parser.add_argument("--data_types", nargs="+", default=None)
parser.add_argument("--task_num_cameras_per_env", type=int, default=1)
parser.add_argument("--ray_caster_visible_mesh_prim_paths", nargs="+", default=["/World/ground"])
parser.add_argument("--height", type=int, default=120)
parser.add_argument("--width", type=int, default=140)
parser.add_argument("--warm_start_length", type=int, default=3)
parser.add_argument("--experiment_length", type=int, default=15)
parser.add_argument("--num_objects", type=int, default=10)
parser.add_argument("--convert_depth_to_camera_to_image_plane", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--autotune", action="store_true")
parser.add_argument("--autotune_max_percentage_util", nargs=4, type=float, default=[100.0, 80.0, 80.0, 80.0])
parser.add_argument("--autotune_max_camera_count", type=int, default=4096)
parser.add_argument("--autotune_camera_count_interval", type=int, default=25)
parser.add_argument("--benchmark_formatter", choices=("json", "osmo", "omniperf", "summary"), default="omniperf")
parser.add_argument("--output_path", type=str, default=".")
AppLauncher.add_app_launcher_args(parser)
# forward unrecognized args as Hydra-style task config overrides
args_cli, hydra_overrides = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.num_cameras < 1:
    parser.error("--num_cameras must be at least 1.")
if args_cli.task_num_cameras_per_env < 1:
    parser.error("--task_num_cameras_per_env must be at least 1.")
if args_cli.autotune:
    import pynvml

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import random
import time
from collections.abc import Sequence
from dataclasses import MISSING

import gymnasium as gym
import numpy as np
import psutil
import torch
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.benchmark import BaseIsaacLabBenchmark, DictMeasurement, SingleMeasurement
from isaaclab.cloner import CloneCfg, ReplicateSession
from isaaclab.sensors import CameraCfg, RayCasterCameraCfg, patterns
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import orthogonalize_perspective_depth, unproject_depth

from isaaclab_tasks.utils import parse_env_cfg


def camera_data_types() -> list[str]:
    """Return explicit outputs for the selected camera implementation."""
    if args_cli.data_types is not None:
        return args_cli.data_types
    return ["rgb", "depth"] if args_cli.camera_type == "camera" else ["distance_to_image_plane"]


def make_camera_cfg(prim_path: str):
    """Declare one camera batch; the clone plan determines its cardinality."""
    if args_cli.camera_type == "ray_caster":
        return RayCasterCameraCfg(
            prim_path=prim_path,
            mesh_prim_paths=args_cli.ray_caster_visible_mesh_prim_paths,
            update_period=0.0,
            offset=RayCasterCameraCfg.OffsetCfg(rot=(1.0, 0.0, 0.0, 0.0)),
            data_types=camera_data_types(),
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                height=args_cli.height,
                width=args_cli.width,
            ),
        )
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=camera_data_types(),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e4),
        ),
        renderer_cfg=IsaacRtxRendererCfg(),
    )


def random_object_cfgs(num_objects: int) -> dict[str, RigidObjectCfg]:
    """Declare the standalone scene's random rigid objects as cfg data."""
    objects = {}
    for index in range(num_objects):
        prim_type = random.choice(("Cube", "Cone", "Cylinder"))
        properties = dict(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(random.random(), random.random(), random.random()), metallic=0.5
            ),
            semantic_tags=[("class", prim_type)],
        )
        if prim_type == "Cube":
            spawn = sim_utils.CuboidCfg(size=(0.25, 0.25, 0.25), **properties)
        elif prim_type == "Cone":
            spawn = sim_utils.ConeCfg(radius=0.1, height=0.25, **properties)
        else:
            spawn = sim_utils.CylinderCfg(radius=0.25, height=0.25, **properties)
        position = (np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])) * np.asarray([1.5, 1.5, 0.5])
        objects[f"object_{index:03d}"] = RigidObjectCfg(
            prim_path=f"/World/Objects/Obj_{index:03d}",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(value) for value in position)),
        )
    return objects


@configclass
class DirectBenchmarkCfg:
    """Complete standalone scene and lifecycle declaration."""

    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(device=args_cli.device, physics=PhysxCfg())
    clone: CloneCfg = CloneCfg()
    ground: AssetBaseCfg = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    objects: dict[str, RigidObjectCfg] = MISSING
    camera: CameraCfg | RayCasterCameraCfg = MISSING


def build_direct_scene(cfg: DirectBenchmarkCfg):
    """Construct every runtime owner inside one cloning lifecycle."""
    with ReplicateSession(
        (cfg.ground, cfg.light, *cfg.objects.values(), cfg.camera),
        num_clones=args_cli.num_cameras,
        env_spacing=0.0,
        clone_strategy=cfg.clone.clone_strategy,
        env_template=cfg.clone.clone_template,
        replicate_physics=cfg.clone.replicate_physics,
    ):
        cfg.ground.class_type(cfg.ground)
        cfg.light.class_type(cfg.light)
        objects = {name: object_cfg.class_type(object_cfg) for name, object_cfg in cfg.objects.items()}
        camera = cfg.camera.class_type(cfg.camera)
    return objects, camera


def inject_cameras_into_task(task: str, num_cameras: int) -> gym.Env:
    """Add explicit camera cfgs to a task before its scene and clone plan are constructed."""
    per_env = args_cli.task_num_cameras_per_env
    if num_cameras % per_env:
        raise ValueError("--num_cameras must be divisible by --task_num_cameras_per_env.")
    cfg = parse_env_cfg(
        task,
        device=args_cli.device,
        num_envs=num_cameras // per_env,
        use_fabric=args_cli.use_fabric,
        overrides=hydra_overrides,
    )
    camera_names = []
    for index in range(per_env):
        name = "benchmark_camera" if index == 0 else f"benchmark_camera_{index}"
        setattr(cfg.scene, name, make_camera_cfg(f"{{ENV_REGEX_NS}}/{name}"))
        camera_names.append(name)
    env = gym.make(task, cfg=cfg)
    env.unwrapped._benchmark_camera_names = tuple(camera_names)
    return env


def utilization(reset: bool = False, maxima: list[float] = [0.0, 0.0, 0.0, 0.0]) -> list[float]:
    """Return maximum CPU, RAM, GPU-compute, and GPU-memory utilization percentages."""
    if reset:
        maxima[:] = [0.0, 0.0, 0.0, 0.0]
    maxima[0] = max(maxima[0], psutil.cpu_percent(interval=0.1))
    maxima[1] = max(maxima[1], psutil.virtual_memory().percent)
    if torch.cuda.is_available() and args_cli.autotune:
        pynvml.nvmlInit()
        for index in range(torch.cuda.device_count()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            maxima[2] = max(maxima[2], pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            maxima[3] = max(maxima[3], 100.0 * memory.used / memory.total)
        pynvml.nvmlShutdown()
    return maxima


def run_simulator(
    cameras: Sequence,
    *,
    sim: sim_utils.SimulationContext | None = None,
    env: gym.Env | None = None,
) -> dict:
    """Run the selected workflow and return timing and utilization measurements."""
    for camera in cameras:
        count = camera.data.intrinsic_matrices.shape[0]
        positions = torch.tensor([[2.5, 2.5, 2.5]], device=camera.device).repeat(count, 1)
        targets = torch.zeros((count, 3), device=camera.device)
        camera.set_world_poses_from_view(positions, targets)

    total_time = 0.0
    sim_step_time = 0.0
    valid_steps = 0
    for step in range(args_cli.experiment_length):
        utilization()
        start = time.perf_counter()
        if sim is not None:
            sim.step()
        else:
            with torch.inference_mode():
                env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))

        for camera in cameras:
            if sim is not None:
                camera.update(dt=sim.get_physics_dt())
            for data_type in camera_data_types():
                output = camera.data.output[data_type]
                if "to" not in data_type and data_type != "depth":
                    continue
                depth = output
                if data_type == "distance_to_camera" and args_cli.convert_depth_to_camera_to_image_plane:
                    depth = orthogonalize_perspective_depth(depth, camera.data.intrinsic_matrices)
                unproject_depth(depth=depth, intrinsics=camera.data.intrinsic_matrices)

        elapsed = time.perf_counter() - start
        sim_step_time += elapsed
        if step > args_cli.warm_start_length:
            total_time += elapsed
            valid_steps += 1

    timing = {
        "average_timestep_duration": total_time / valid_steps if valid_steps else 0.0,
        "average_sim_step_duration": sim_step_time / args_cli.experiment_length,
        "total_simulation_time": sim_step_time,
    }
    system = utilization()
    print(f"Average timestep: {timing['average_timestep_duration']:.6f} s")
    print(f"Average simulation step: {timing['average_sim_step_duration']:.6f} s")
    print(f"CPU {system[0]:.1f}% | RAM {system[1]:.1f}% | GPU {system[2]:.1f}% | VRAM {system[3]:.1f}%")
    return {"timing_analytics": timing, "system_utilization_analytics": system}


def record_results(benchmark: BaseIsaacLabBenchmark, analysis: dict) -> None:
    """Record one benchmark result."""
    timing = analysis["timing_analytics"]
    for name, key in (
        ("Average Timestep Duration", "average_timestep_duration"),
        ("Average Simulation Step Duration", "average_sim_step_duration"),
        ("Total Simulation Time", "total_simulation_time"),
    ):
        benchmark.add_measurement(
            "runtime", measurement=SingleMeasurement(name=name, value=timing[key] * 1000.0, unit="ms")
        )
    system = analysis["system_utilization_analytics"]
    benchmark.add_measurement(
        "runtime",
        measurement=DictMeasurement(
            name="System Utilization",
            value=dict(
                cpu_percent=system[0],
                ram_percent=system[1],
                gpu_compute_percent=system[2],
                gpu_memory_percent=system[3],
            ),
        ),
    )


def main() -> None:
    """Run the direct scene once or autotune task camera count."""
    benchmark = BaseIsaacLabBenchmark(
        benchmark_name="benchmark_cameras",
        formatter_type=args_cli.benchmark_formatter,
        output_path=args_cli.output_path,
        use_recorders=True,
        frametime_recorders=args_cli.benchmark_formatter in ("summary", "omniperf"),
        output_prefix="benchmark_cameras",
        workflow_metadata={
            "metadata": [
                {"name": "task", "data": args_cli.task},
                {"name": "camera_type", "data": args_cli.camera_type},
                {"name": "num_cameras", "data": args_cli.num_cameras},
                {"name": "height", "data": args_cli.height},
                {"name": "width", "data": args_cli.width},
            ]
        },
    )

    if args_cli.task is None:
        cfg = DirectBenchmarkCfg(
            objects=random_object_cfgs(args_cli.num_objects),
            camera=make_camera_cfg("{ENV_REGEX_NS}/BenchmarkCamera"),
        )
        sim = sim_utils.SimulationContext(cfg.sim)
        _objects, camera = build_direct_scene(cfg)
        sim.reset()
        analysis = run_simulator([camera], sim=sim)
    else:
        camera_count = args_cli.num_cameras
        analysis = None
        while camera_count <= args_cli.autotune_max_camera_count:
            env = inject_cameras_into_task(args_cli.task, camera_count)
            env.reset()
            cameras = [env.unwrapped.scene[name] for name in env.unwrapped._benchmark_camera_names]
            analysis = run_simulator(cameras, env=env)
            within_limits = all(
                value <= limit
                for value, limit in zip(
                    analysis["system_utilization_analytics"], args_cli.autotune_max_percentage_util, strict=True
                )
            )
            env.close()
            if not args_cli.autotune or not within_limits:
                break
            camera_count += args_cli.autotune_camera_count_interval
        assert analysis is not None

    record_results(benchmark, analysis)
    benchmark.update_manual_recorders()
    benchmark.finalize()


if __name__ == "__main__":
    main()
    simulation_app.close()
