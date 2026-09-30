# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose a curated selection of task scenes into one heterogeneous simulation.

The pipeline is: resolve the scene config of every task in :data:`DEFAULT_TASKS`,
declare one clone combination per task while skipping every task's own light and
floor, add one Dome light and one shared ground plane, and clone the composition
so each environment hosts one task's assets. No task environments or MDP managers
are constructed; the demo owns generic PhysX simulation settings.

.. code-block:: bash

    # Usage with the full default task selection.
    ./isaaclab.sh -p scripts/demos/heterogeneous_scene.py

    # Usage with a smaller composition.
    ./isaaclab.sh -p scripts/demos/heterogeneous_scene.py --num_task 3 --num_envs 3

"""

from __future__ import annotations

import argparse
from dataclasses import MISSING

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import preset, resolve_config, resolve_task_config, setup_preset_cli
from isaaclab_tasks.utils.presets import MultiBackendCameraCfg, MultiBackendSimulationCfg

parser = argparse.ArgumentParser(
    description="Demo: clone-only multi-robot multi-task scene.",
    conflict_handler="resolve",
)
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--env_spacing", type=float, default=2.5, help="Distance between environment origins [m].")
parser.add_argument("--sim_dt", type=float, default=1.0 / 60.0, help="Physics timestep [s].")
parser.add_argument(
    "--num_task",
    type=int,
    default=None,
    help="Number of tasks to use from the default order. Omit to use all tasks.",
)
add_launcher_args(parser)
args_cli, config_overrides = setup_preset_cli(parser)

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import InclusionSet
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import find_unique_string_name
from isaaclab.utils.configclass import configclass

from isaaclab_physx.physics import PhysxCfg  # isort: skip


@configclass
class DemoCfg:
    """Heterogeneous-scene demo configuration."""

    sim: MultiBackendSimulationCfg = MultiBackendSimulationCfg(
        dt=args_cli.sim_dt,
        device=args_cli.device,
        physics=preset(default=PhysxCfg(), isaacsim_physx=PhysxCfg()),
    )
    scene: InteractiveSceneCfg = MISSING


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


def _load_task_scenes() -> tuple[list[str], list[InteractiveSceneCfg]]:
    """Resolve the scene config of every selected task."""
    task_ids = list(DEFAULT_TASKS if args_cli.num_task is None else DEFAULT_TASKS[: args_cli.num_task])
    if len(task_ids) < 2:
        raise ValueError("Select at least two task scenes.")
    scene_cfgs = []
    for task_id in task_ids:
        env_cfg, _ = resolve_task_config(task_id, None, overrides=config_overrides)
        scene_cfgs.append(env_cfg.scene)
    return task_ids, scene_cfgs


def _compose_task_scenes(scene_cfgs: list[InteractiveSceneCfg]) -> InteractiveSceneCfg:
    """Declare the selected task assets as clone combinations on the first scene config."""
    base_fields = InteractiveSceneCfg.__dataclass_fields__

    def environment_assets(scene_cfg: InteractiveSceneCfg) -> list[tuple[str, AssetBaseCfg]]:
        return [
            (name, value)
            for name, value in vars(scene_cfg).items()
            if name not in base_fields
            and isinstance(value, AssetBaseCfg)
            and value.spawn is not None
            and not isinstance(value.spawn, (sim_utils.LightCfg, sim_utils.GroundPlaneCfg))
        ]

    target = scene_cfgs[0]
    target_assets = environment_assets(target)
    target_names = {name for name, _ in target_assets}
    for name, value in vars(target).items():
        if name not in base_fields and value is not None and name not in target_names:
            setattr(target, name, None)

    combinations = [InclusionSet(assets=[name for name, _ in target_assets])]
    used_names, paths = set(dir(target)), {asset.prim_path for _, asset in target_assets}
    for source in scene_cfgs[1:]:
        available, added_names = dict(target_assets), []
        for source_name, asset in environment_assets(source):
            target_name = next((name for name, existing in available.items() if existing == asset), None)
            if target_name is not None:
                del available[target_name]
            else:
                target_name = find_unique_string_name(source_name, lambda name: name not in used_names)
                if asset.prim_path in paths:
                    asset = asset.replace(
                        prim_path=find_unique_string_name(asset.prim_path, lambda path: path not in paths)
                    )
                setattr(target, target_name, asset)
                target_assets.append((target_name, asset))
                used_names.add(target_name)
                paths.add(asset.prim_path)
            added_names.append(target_name)
        combinations.append(InclusionSet(assets=list(dict.fromkeys(added_names))))
    target.clone_cfg.clone_combinations = combinations
    return target


def main() -> None:
    """Resolve the selected task scenes, compose, add light and floor, simulate."""
    # Resolve and compose every task scene before Kit launches: config resolution is
    # simulator-free, and the launch swaps module state that must not interleave with it.
    task_ids, task_scene_cfgs = _load_task_scenes()
    print(f"\n[INFO] Composing task scenes: {task_ids}")
    for task_scene_cfg in task_scene_cfgs:
        task_scene_cfg.env_spacing = args_cli.env_spacing

    scene_cfg = _compose_task_scenes(task_scene_cfgs)
    scene_cfg.light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    scene_cfg.ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    scene_cfg.camera = MultiBackendCameraCfg()

    scene_cfg.num_envs = args_cli.num_envs
    scene_cfg.replicate_physics = True

    cfg = resolve_config(DemoCfg(scene=scene_cfg), config_overrides)
    with launch_simulation(cfg, args_cli):
        sim = sim_utils.SimulationContext(cfg.sim)
        sim.set_camera_view(eye=[6.0, 6.0, 4.0], target=[0.0, 0.0, 0.5])
        scene = cfg.scene.class_type(cfg.scene)
        sim.reset()
        scene.reset()
        scene.write_data_to_sim()
        print(f"[INFO] Composed {len(task_ids)} task scenes into {args_cli.num_envs} environments. Stepping physics.")

        sim_dt = sim.get_physics_dt()
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
