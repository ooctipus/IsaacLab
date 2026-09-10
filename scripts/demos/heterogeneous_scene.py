# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose a curated selection of task scenes into one heterogeneous simulation.

The pipeline is: resolve the scene config of every task in :data:`DEFAULT_TASKS`,
fold the scenes together with :func:`~isaaclab.scene.add` while skipping every
task's own light and floor, add one Dome light and one shared ground plane, and
clone the composition so each environment hosts one task's assets. No task
environments or MDP managers are constructed; the demo owns generic PhysX
simulation settings.

.. code-block:: bash

    # FeatherPGS with the full supported task selection.
    uv run python scripts/demos/heterogeneous_scene.py --physics feather_pgs --visualizer none

    # Usage with a smaller composition.
    ./isaaclab.sh -p scripts/demos/heterogeneous_scene.py --num_task 3 --num_envs 3

"""

from __future__ import annotations

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
import os
import statistics
import sys
import time

# Concurrent USD spawning of heterogeneous assets is unstable in pxr's worker pool.
os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="Demo: clone-only multi-robot multi-task scene.",
    conflict_handler="resolve",
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="Distance between environment origins [m].")
parser.add_argument(
    "--sim_dt",
    type=float,
    default=1.0 / 200.0,
    help="Physics timestep [s]. The composed locomotion scenes are authored for 200 Hz.",
)
parser.add_argument(
    "--num_task",
    type=int,
    default=None,
    help="Number of tasks to use from the default order. Omit to use all tasks.",
)
parser.add_argument(
    "--task",
    default=None,
    help="Clone one supported task into every environment instead of composing heterogeneous tasks.",
)
parser.add_argument(
    "--benchmark_steps",
    type=int,
    default=None,
    help="Synchronously time this many steps after warmup, print world-steps/s, and exit.",
)
parser.add_argument("--benchmark_repeats", type=int, default=5, help="Number of synchronized benchmark repeats.")
parser.add_argument("--warmup_steps", type=int, default=200, help="Steps excluded from a finite benchmark.")
parser.add_argument(
    "--physics", default="isaacsim_physx", choices=["isaacsim_physx", "feather_pgs"], help="Physics backend."
)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = parser.parse_known_args()
# strip consumed args so hydra-based task-config resolution does not re-parse them
sys.argv = [sys.argv[0], *hydra_args]

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.scene import add as scene_add

from isaaclab_tasks.utils import resolve_task_config

# Tasks composed by default. The selection criterion is simple: every listed
# scene is a PhysX task whose floor is a single flat plane at height zero, so
# one shared ground plane can serve the whole composition. Registered tasks
# not listed here either place their floor elsewhere (e.g. tabletop scenes
# with the ground at -1.05 m), use procedural terrain (the -Rough velocity
# tasks), require camera rendering flags or optional packages, target the
# Newton backend, or duplicate a listed task under another alias.
DEFAULT_TASKS = (
    # classic control
    "Isaac-Cartpole",
    "Isaac-Fourbar-Pole-Swingup",
    "Isaac-Ant",
    "Isaac-Humanoid",
    # legged locomotion
    "Isaac-Velocity-Flat-AnymalD",
    "IsaacContrib-Velocity-Flat-AnymalB",
    "IsaacContrib-Velocity-Flat-AnymalC",
    "IsaacContrib-Velocity-Flat-UnitreeA1",
    "IsaacContrib-Velocity-Flat-UnitreeGo1",
    "Isaac-Velocity-Flat-UnitreeGo2",
    "Isaac-Velocity-Flat-Cassie",
    "IsaacContrib-Velocity-Flat-Digit",
    "Isaac-Velocity-Flat-G1",
    "Isaac-Velocity-Flat-H1",
    "IsaacContrib-Navigation-Flat-AnymalC",
    # arm and hand manipulation
    "Isaac-Lift-Franka",
    "Isaac-Reorient-Franka",
    "Isaac-Lift-KukaAllegro",
    "Isaac-Reorient-KukaAllegro",
    "Isaac-Open-Drawer-Franka",
    "IsaacContrib-Open-Drawer-Franka-IK-Abs",
    "IsaacContrib-Open-Drawer-Franka-IK-Rel",
)

# Every scene here exposes a FeatherPGS physics preset and uses a shared flat
# ground plane. Fixed-tendon and camera tasks are intentionally excluded.
FEATHER_PGS_TASKS = (
    "Isaac-Cartpole",
    "Isaac-Ant",
    "Isaac-Humanoid",
    "Isaac-Velocity-Flat-AnymalD",
    "IsaacContrib-Velocity-Flat-AnymalB",
    "IsaacContrib-Velocity-Flat-AnymalC",
    "IsaacContrib-Velocity-Flat-UnitreeA1",
    "IsaacContrib-Velocity-Flat-UnitreeGo1",
    "Isaac-Velocity-Flat-UnitreeGo2",
    "Isaac-Velocity-Flat-Cassie",
    "Isaac-Velocity-Flat-G1",
    "Isaac-Velocity-Flat-H1",
    "Isaac-Lift-Franka",
    "Isaac-Reorient-Franka",
    "Isaac-Lift-KukaAllegro",
    "Isaac-Reorient-KukaAllegro",
    "Isaac-Open-Drawer-Franka",
    "IsaacContrib-Open-Drawer-Franka-IK-Abs",
    "IsaacContrib-Open-Drawer-Franka-IK-Rel",
)


def _load_task_scenes() -> tuple[list[str], list[InteractiveSceneCfg]]:
    """Resolve the scene config of every selected task."""
    available_tasks = FEATHER_PGS_TASKS if args_cli.physics == "feather_pgs" else DEFAULT_TASKS
    if args_cli.task is not None and args_cli.num_task is not None:
        raise ValueError("--task and --num_task are mutually exclusive.")
    if args_cli.task is not None:
        if args_cli.task not in available_tasks:
            raise ValueError(f"Unsupported task {args_cli.task!r}; choose one of {available_tasks}.")
        task_ids = [args_cli.task]
    else:
        task_ids = list(available_tasks if args_cli.num_task is None else available_tasks[: args_cli.num_task])
    if args_cli.task is None and len(task_ids) < 2:
        raise ValueError("Select at least two task scenes.")
    scene_cfgs = []
    for task_id in task_ids:
        overrides = [f"physics={args_cli.physics}", *hydra_args]
        env_cfg, _ = resolve_task_config(task_id, "", overrides=overrides)
        scene_cfgs.append(env_cfg.scene)
    return task_ids, scene_cfgs


def main() -> None:
    """Resolve the selected task scenes, compose, add light and floor, simulate."""
    if args_cli.benchmark_steps is not None and args_cli.benchmark_steps <= 0:
        raise ValueError("benchmark_steps must be positive.")
    if args_cli.benchmark_repeats <= 0:
        raise ValueError("benchmark_repeats must be positive.")
    if args_cli.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    # Resolve and compose every task scene before Kit launches: config resolution is
    # simulator-free, and the launch swaps module state that must not interleave with it.
    task_ids, task_scene_cfgs = _load_task_scenes()
    print(f"\n[INFO] Composing task scenes: {task_ids}")

    def is_global_asset(a: AssetBaseCfg) -> bool:
        return isinstance(a.spawn, (sim_utils.LightCfg, sim_utils.GroundPlaneCfg))

    base_fields = InteractiveSceneCfg.__dataclass_fields__
    for task_scene_cfg in task_scene_cfgs:
        task_scene_cfg.env_spacing = args_cli.env_spacing

    scene_cfg = task_scene_cfgs[0]
    if len(task_scene_cfgs) > 1:
        for task_scene_cfg in task_scene_cfgs:
            for asset_name, asset_cfg in vars(task_scene_cfg).items():
                if asset_name not in base_fields and isinstance(asset_cfg, AssetBaseCfg) and is_global_asset(asset_cfg):
                    setattr(task_scene_cfg, asset_name, None)
        for task_scene_cfg in task_scene_cfgs[1:]:
            scene_cfg = scene_add(scene_cfg, task_scene_cfg)
        scene_cfg.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        scene_cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    scene_cfg.num_envs = args_cli.num_envs
    scene_cfg.replicate_physics = True

    if args_cli.physics == "feather_pgs":
        # This demo owns no per-asset state or controls. Avoid constructing sparse
        # per-type views and let one Newton model own the heterogeneous world domain.
        for asset_name, asset_cfg in vars(scene_cfg).items():
            if asset_name not in base_fields and isinstance(asset_cfg, AssetBaseCfg):
                asset_cfg.class_type = None

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=args_cli.sim_dt,
                device=args_cli.device,
                physics=physics_cfg,
                use_newton_actuators=args_cli.physics == "feather_pgs",
            )
        )
        sim.set_camera_view(eye=[6.0, 6.0, 4.0], target=[0.0, 0.0, 0.5])
        scene = scene_cfg.class_type(scene_cfg)
        sim.reset()
        scene.reset()
        scene.write_data_to_sim()
        print(f"[INFO] Composed {len(task_ids)} task scenes into {args_cli.num_envs} environments. Stepping physics.")

        sim_dt = sim.get_physics_dt()

        if args_cli.benchmark_steps is not None:
            import warp as wp

            if args_cli.physics == "feather_pgs":
                from isaaclab_newton.physics import NewtonManager

                model = NewtonManager.get_model()
                print(
                    "[MODEL] "
                    f"worlds={model.world_count} bodies={model.body_count} shapes={model.shape_count} "
                    f"joints={model.joint_count} joint_dofs={model.joint_dof_count} actuators={len(model.actuators)}"
                )

            for _ in range(args_cli.warmup_steps):
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
            repeat_fps = []
            for repeat in range(args_cli.benchmark_repeats):
                wp.synchronize_device(args_cli.device)
                start = time.perf_counter()
                for _ in range(args_cli.benchmark_steps):
                    scene.write_data_to_sim()
                    sim.step()
                    scene.update(sim_dt)
                wp.synchronize_device(args_cli.device)
                elapsed = time.perf_counter() - start
                fps = args_cli.num_envs * args_cli.benchmark_steps / elapsed
                repeat_fps.append(fps)
                print(
                    f"[RESULT] repeat={repeat + 1}/{args_cli.benchmark_repeats} elapsed_s={elapsed:.6f} fps={fps:.3f}"
                )
            solver_cfg = getattr(physics_cfg, "solver_cfg", None)
            fps_std = statistics.stdev(repeat_fps) if len(repeat_fps) > 1 else 0.0
            print(
                "[SUMMARY] "
                f"physics={args_cli.physics} num_envs={args_cli.num_envs} tasks={len(task_ids)} "
                f"composition={args_cli.task or 'heterogeneous'} "
                f"dt={sim_dt:.9f} steps={args_cli.benchmark_steps} repeats={args_cli.benchmark_repeats} "
                f"fps_mean={statistics.fmean(repeat_fps):.3f} fps_median={statistics.median(repeat_fps):.3f} "
                f"fps_std={fps_std:.3f} "
                f"num_substeps={getattr(physics_cfg, 'num_substeps', None)} "
                f"solver={type(solver_cfg).__name__} "
                f"mode={getattr(solver_cfg, 'pgs_mode', None)} "
                f"iterations={getattr(solver_cfg, 'pgs_iterations', None)} "
                f"cuda_graph={getattr(physics_cfg, 'use_cuda_graph', None)}"
            )
            return

        # Step while a visualizer window is still open (or none exist, e.g. headless).
        while True:
            if not sim.is_playing():
                sim.step()
                continue
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)


if __name__ == "__main__":
    main()
